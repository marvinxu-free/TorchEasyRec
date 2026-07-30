# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from datetime import timedelta
from queue import Queue
from typing import Callable, Iterator, List, Optional, Tuple, Type

import torch
from torch import distributed as dist
from torch import nn
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding_types import (
    KJTList,
)
from torchrec.distributed.embeddingbag import (
    ShardedEmbeddingBagCollection,
    _create_mean_pooling_divisor,
)
from torchrec.distributed.mc_embedding_modules import (
    ShrdCtx,
)
from torchrec.distributed.mc_embeddingbag import (
    ShardedManagedCollisionEmbeddingBagCollection,
)
from torchrec.distributed.model_parallel import DataParallelWrapper
from torchrec.distributed.model_parallel import (
    DistributedModelParallel as _DistributedModelParallel,
)
from torchrec.distributed.train_pipeline import TrainPipeline, TrainPipelineContext
from torchrec.distributed.train_pipeline import TrainPipelineBase as _TrainPipelineBase
from torchrec.distributed.train_pipeline import (
    TrainPipelineSparseDist as _TrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.pipeline_context import In, Out
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ShardedModule,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from tzrec.utils.logging_util import logger


def init_process_group() -> Tuple[torch.device, str]:
    """Init process_group, device, rank, backend."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    pg_timeout = None
    if "PROCESS_GROUP_TIMEOUT_SECONDS" in os.environ:
        pg_timeout = timedelta(
            seconds=(int(os.environ["PROCESS_GROUP_TIMEOUT_SECONDS"]))
        )
    dist.init_process_group(backend=backend, timeout=pg_timeout)

    return device, backend


def get_dist_object_pg(world_size: Optional[int] = None) -> Optional[dist.ProcessGroup]:
    """New ProcessGroup used for broadcast_object or gather_object."""
    pg = None
    world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # pyre-ignore [16]
        if dist.is_initialized() and dist.GroupMember.WORLD.size() == world_size:
            pg = dist.GroupMember.WORLD
        else:
            pg = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    return pg


# fix missing create_mean_pooling_callback of mc-ebc input_dist
def _mc_ebc_input_dist(
    # pyre-ignore [2]
    self,
    ctx: ShrdCtx,
    features: KeyedJaggedTensor,
) -> Awaitable[Awaitable[KJTList]]:
    ctx.variable_batch_per_feature = features.variable_stride_per_key()
    ctx.inverse_indices = features.inverse_indices_or_none()

    if self._embedding_module._has_uninitialized_input_dist:
        self._features_order = []
        # disable feature permutation in mc, because we should
        # permute features in mc-ebc before mean pooling callback.
        if self._managed_collision_collection._has_uninitialized_input_dists:
            self._managed_collision_collection._create_input_dists(
                input_feature_names=features.keys()
            )
            self._managed_collision_collection._has_uninitialized_input_dists = False
            if self._managed_collision_collection._features_order:
                self._features_order = (
                    self._managed_collision_collection._features_order
                )
                self._managed_collision_collection._features_order = []
            # merge vbe support for mc-ebc in torchrec 1.5.0
            if ctx.variable_batch_per_feature:
                if self._return_remapped_features:
                    raise NotImplementedError(
                        "VBE is not supported currently for "
                        "return_remapped_features=True."
                    )
                # pyre-ignore [16]
                self._embedding_module._create_inverse_indices_permute_indices(
                    ctx.inverse_indices
                )
        if self._embedding_module._has_mean_pooling_callback:
            self._embedding_module._init_mean_pooling_callback(
                features.keys(),
                ctx.inverse_indices,
            )
        self._embedding_module._has_uninitialized_input_dist = False

    with torch.no_grad():
        if self._features_order:
            features = features.permute(
                self._features_order,
                self._managed_collision_collection._features_order_tensor,
            )
        if self._embedding_module._has_mean_pooling_callback:
            ctx.divisor = _create_mean_pooling_divisor(
                lengths=features.lengths(),
                stride=features.stride(),
                keys=features.keys(),
                offsets=features.offsets(),
                pooling_type_to_rs_features=self._embedding_module._pooling_type_to_rs_features,
                stride_per_key=features.stride_per_key(),
                dim_per_key=self._embedding_module._dim_per_key,
                embedding_names=self._embedding_module._embedding_names,
                embedding_dims=self._embedding_module._embedding_dims,
                variable_batch_per_feature=ctx.variable_batch_per_feature,
                kjt_inverse_order=self._embedding_module._kjt_inverse_order,
                kjt_key_indices=self._embedding_module._kjt_key_indices,
                kt_key_ordering=self._embedding_module._kt_key_ordering,
                inverse_indices=ctx.inverse_indices,
                weights=features.weights_or_none(),
            )
    # TODO: resolve incompatibility with different contexts
    return self._managed_collision_collection.input_dist(ctx, features)


ShardedManagedCollisionEmbeddingBagCollection.input_dist = _mc_ebc_input_dist


def DistributedModelParallel(
    module: nn.Module,
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    init_data_parallel: bool = True,
    init_parameters: bool = True,
    data_parallel_wrapper: Optional[DataParallelWrapper] = None,
) -> _DistributedModelParallel:
    """Entry point to model parallelism.

    we custom ddp to make input_dist of ShardModel uninitialized.
    mc-ebc now make _has_uninitialized_input_dist = True in init.
    TODO: use torchrec DistributedModelParallel when torchrec fix it.
    """
    model = _DistributedModelParallel(
        module,
        env,
        device,
        plan,
        sharders,
        init_data_parallel,
        init_parameters,
        data_parallel_wrapper,
    )
    for _, m in model.named_modules():
        if hasattr(m, "_has_uninitialized_input_dist") and isinstance(
            m, ShardedEmbeddingBagCollection
        ):
            m._has_uninitialized_input_dist = True
    return model


def _pipeline_backward(losses: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
    with record_function("## backward ##"):
        loss = torch.sum(losses, dim=0)
        if (
            hasattr(optimizer, "_gradient_accumulation_steps")
            # pyre-ignore [16]
            and optimizer._gradient_accumulation_steps > 1
        ):
            loss = loss / optimizer._gradient_accumulation_steps
        # pyre-ignore [16]
        if hasattr(optimizer, "_grad_scaler") and optimizer._grad_scaler is not None:
            optimizer._grad_scaler.scale(loss).backward()
        else:
            loss.backward()


class TrainPipelineBase(_TrainPipelineBase):
    """TorchEasyRec's TrainPipelineBase, make backward support grad scaler."""

    def _backward(self, losses: torch.Tensor) -> None:
        _pipeline_backward(losses, self._optimizer)


def _pcgrad_backward(
    losses: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    dense_params: List[torch.nn.Parameter],
) -> None:
    """PCGrad backward: project per-task dense grads, keep sparse grads standard.

    Contract: ``losses`` is a 1-D tensor with one scalar per task (produced by
    ``TrainWrapper`` when ``model._use_pcgrad`` is True).

    Steps:
        1. Extract per-task dense grads via ``torch.autograd.grad`` (does NOT
           fire the fused in-backward sparse optimizer nor DDP grad-sync hooks).
        2. PCGrad-project the per-task dense grads to remove conflicts.
        3. Run the real-graph ``backward`` on the summed loss: this populates
           sparse embedding grads (standard sum) and fires the fused sparse
           optimizer in-backward, plus a single DDP all-reduce.
        4. Overwrite each dense param's ``.grad`` with the projected+summed
           target (scaled to match the GradScaler), so the subsequent
           ``optimizer.step()`` steps dense params with PCGrad gradients.
    """
    from tzrec.loss.pcgrad import (
        flatten_grads_with_zeros,
        pcgrad_project,
        write_grads,
    )

    with record_function("## pcgrad backward ##"):
        assert losses.dim() == 1, (
            f"PCGrad expects a 1-D per-task loss tensor, got shape {losses.shape}"
        )
        accum = getattr(optimizer, "_gradient_accumulation_steps", 0) or 0
        assert accum <= 1, (
            "PCGrad does not support gradient_accumulation_steps > 1 yet; "
            "set train_config.gradient_accumulation_steps = 0 or 1 to use pcgrad."
        )
        scaler = getattr(optimizer, "_grad_scaler", None)

        # Only params that require grad can be differentiated; frozen params
        # (requires_grad=False) would raise inside autograd.grad, so filter
        # them out. They have no gradient to project either way.
        dense_params = [p for p in dense_params if p.requires_grad]
        zero_flat = (
            torch.zeros(
                sum(p.numel() for p in dense_params),
                device=losses.device,
            )
            if dense_params
            else None
        )

        # losses layout (see TrainWrapper.forward): indices [:-1] are main
        # task tower losses to PCGrad-project; the LAST element is the summed
        # auxiliary loss (bias heads + dropout regularizers) whose dense grad
        # is added to the projected target WITHOUT projection.
        main_losses = losses[:-1]
        aux_loss = losses[-1]

        # 1. per-main-task dense grads (unscaled, real graph retained for backward)
        per_task_flat = []
        for i in range(main_losses.shape[0]):
            # A non-grad loss entry (e.g. a constant term) contributes nothing;
            # skip autograd.grad for it (it would otherwise raise) and use zeros.
            if not main_losses[i].requires_grad or not dense_params:
                per_task_flat.append(zero_flat)
                continue
            grads = torch.autograd.grad(
                main_losses[i],
                dense_params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            per_task_flat.append(flatten_grads_with_zeros(grads, dense_params))

        # 2. project + sum the MAIN task grads
        proj = pcgrad_project(per_task_flat)
        target_dense = torch.stack(proj).sum(dim=0)

        # 2b. add the aux loss's dense grad WITHOUT projection. The real-graph
        #     backward below (torch.sum over all elements) drives sparse params
        #     with aux's gradient; but step 4 overwrites dense .grad, so the
        #     aux dense grad must be folded into target_dense to survive.
        if aux_loss.requires_grad and dense_params:
            aux_grads = torch.autograd.grad(
                aux_loss,
                dense_params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            target_dense = target_dense + flatten_grads_with_zeros(
                aux_grads, dense_params
            )

        # 3. real-graph backward for sparse params + DDP sync. torch.sum over
        #    all elements includes the aux loss, so sparse params receive the
        #    aux gradient (correct: aux is a regularizer that should train them).
        loss = torch.sum(losses, dim=0)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 4. overwrite dense .grad with projected target (scaler-scaled so the
        #    scaler's upcoming unscale_ divides it back to the true magnitude).
        #
        # IMPORTANT (multi-GPU correctness): the per-task grads above were
        # extracted via ``torch.autograd.grad``, which does NOT trigger DDP
        # gradient sync. The real-graph ``backward`` in step 3 did fire DDP
        # all-reduce on the dense grads, but we are about to overwrite that
        # result with ``target_dense`` — which is still LOCAL to this rank.
        # Without an explicit all-reduce here, each rank would step its dense
        # params on its own local-batch projected gradient, breaking DDP's
        # "all ranks apply the same averaged update" invariant: effective
        # batch size degrades from world_size*B to B and dense params drift
        # apart across ranks. All-reduce the projected target so dense grads
        # get the same global averaging that sparse grads (via step 3) get.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(target_dense, op=dist.ReduceOp.SUM)
            target_dense = target_dense / dist.get_world_size()
        scale = scaler.get_scale() if scaler is not None else 1.0
        write_grads(dense_params, target_dense, scale=scale)


class PCGradMixin:
    """Mixin enabling PCGrad backward; set ``self._dense_params`` before use."""

    _dense_params: List[torch.nn.Parameter] = []

    def set_dense_params(self, params: List[torch.nn.Parameter]) -> None:
        """Inject the list of dense (non-fused-sparse) parameters to project."""
        self._dense_params = params

    def _backward(self, losses: torch.Tensor) -> None:
        _pcgrad_backward(losses, self._optimizer, self._dense_params)


class TrainPipelinePCGradBase(PCGradMixin, TrainPipelineBase):
    """PCGrad variant of TrainPipelineBase (no sparse modules)."""


class TrainPipelineSparseDist(_TrainPipelineSparseDist):
    """TorchEasyRec's TrainPipelineSparseDist, make backward support grad scaler."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enqueue_batch_after_forward: bool = False,
        check_all_workers_data_status: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            context_type,
            pipeline_postproc,
            custom_model_fwd,
            dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward,
        )
        self._check_all_workers_data_status = check_all_workers_data_status
        self._sync_at_progress_entry = (
            device.type == "cuda"
            and dist.is_initialized()
            and torch.cuda.get_device_capability(device)[0] < 7
        )
        if self._sync_at_progress_entry and int(os.environ.get("RANK", 0)) == 0:
            logger.info(
                "TrainPipelineSparseDist: cc<7.0 detected on %s; "
                "enabling per-iter dist.barrier() at progress() entry",
                device,
            )

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """Run one training iter, with optional pre-iter barrier on cc<7.0."""
        if self._sync_at_progress_entry:
            # workaround NCCL deadlock from unconditional next-batch
            # KJT a2a in upstream progress() on Pascal GPUs (cc<7.0).
            dist.barrier()
        return super().progress(dataloader_iter)

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)

            if self._check_all_workers_data_status:
                # Check if all workers either have or do not have a batch available.
                has_batch = torch.tensor(
                    0 if batch is None else 1, dtype=torch.float, device=self._device
                )
                dist.all_reduce(has_batch, dist.ReduceOp.AVG)
                if has_batch.item() < 1:
                    # We drop remainder batches on all workers,
                    # if one worker does not have a batch
                    self._dataloader_exhausted = True
                    batch = None
            else:
                if batch is None:
                    self._dataloader_exhausted = True
        return batch

    def _backward(self, losses: torch.Tensor) -> None:
        _pipeline_backward(losses, self._optimizer)


class TrainPipelinePCGradSparseDist(PCGradMixin, TrainPipelineSparseDist):
    """PCGrad variant of TrainPipelineSparseDist.

    Defined after :class:`TrainPipelineSparseDist` since it subclasses it.
    """


class PredictPipelineSparseDist(_TrainPipelineSparseDist):
    """TorchEasyRec's PredictPipelineSparseDist, make predict do not hang."""

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)

            # Check if all workers either have or do not have a batch available.
            has_batch = torch.tensor(
                0 if batch is None else 1, dtype=torch.float, device=self._device
            )
            dist.all_reduce(has_batch, dist.ReduceOp.AVG)
            if batch is None:
                if has_batch.item() > 0:
                    # If some workers still have a batch, create a dummy batch
                    # to avoid potential hang.
                    batch = copy.copy(self.batches[0])
                    batch.dummy = True
                else:
                    self._dataloader_exhausted = True
        return batch


def create_train_pipeline(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    check_all_workers_data_status: bool = False,
    use_pcgrad: bool = False,
    dense_params: Optional[List[torch.nn.Parameter]] = None,
) -> TrainPipeline:
    """Create TrainPipeline.

    Args:
        model (nn.Module): a DMP model.
        optimizer (torch.optim.Optimizer): a KeyedOptimizer.
        check_all_workers_data_status (bool): check data on all workers
            is available or not.
        use_pcgrad (bool): when True, use a TrainPipeline variant whose backward
            applies PCGrad over per-task dense gradients.
        dense_params (list, optional): dense (non-fused-sparse) parameters
            required by the PCGrad backward path.

    Return:
        a TrainPipeline.
    """
    has_sparse_module = False

    q = Queue()
    q.put(model.module)
    while not q.empty():
        m = q.get()
        if isinstance(m, ShardedModule):
            has_sparse_module = True
            break
        else:
            for child in m.children():
                q.put(child)

    if not has_sparse_module:
        # use TrainPipelineBase when model do not have sparse parameters.
        if use_pcgrad:
            pipeline = TrainPipelinePCGradBase(model, optimizer, model.device)
            pipeline.set_dense_params(dense_params or [])
            return pipeline
        # pyre-ignore [6]
        return TrainPipelineBase(model, optimizer, model.device)
    else:
        if use_pcgrad:
            pipeline = TrainPipelinePCGradSparseDist(
                model,
                # pyre-ignore [6]
                optimizer,
                model.device,
                execute_all_batches=True,
                check_all_workers_data_status=check_all_workers_data_status,
            )
            pipeline.set_dense_params(dense_params or [])
            return pipeline
        return TrainPipelineSparseDist(
            model,
            # pyre-ignore [6]
            optimizer,
            model.device,
            execute_all_batches=True,
            check_all_workers_data_status=check_all_workers_data_status,
        )
