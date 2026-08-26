# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.weighted_infonce import weighted_infonce_loss
from tzrec.models.mtl_rocket_launching2 import MTLRocketLaunching2
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.modules.ns_gate import NSGate
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _wcl_for_task(
    input_fea: torch.Tensor,  # [B, d]  input feature vectors x (embedding
    # concat, pre share_mlp; L2-normalized by caller)
    logits: torch.Tensor,  # [B]
    label: torch.Tensor,  # [B], float (1 positive, 0 negative)
    num_negatives: int,
    temperature: float,
    delta: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """In-batch weighted contrastive loss for one task (SDCL Eq.7 + Eq.8).

    ``@torch.fx.wrap`` makes this an opaque leaf for ``torch.fx`` symbolic
    tracing (used by TorchRec's train pipeline, which traces the full
    ``TrainWrapper.forward`` including ``loss()``). The body relies on
    data-dependent control flow -- ``.item()`` / ``torch.nonzero`` /
    ``torch.randint`` with a data-dependent upper bound -- that FX cannot
    trace. Wrapping lets the tracer record a single ``call_function`` node and
    skip the body, which then runs only at eager (training) time. Same pattern
    as ``tzrec/loss/jrc_loss.py`` and ``tzrec/models/dat.py``.

    Semantics: in-batch positives are ``label == 1``, the negative pool is
    ``label == 0``; each positive draws its OWN ``num_negatives`` negatives
    (per-positive sampling, 2026-08-20 -- previously N negatives were shared
    by all P positives, which concentrated a gradient of scale ``α·P/τ``
    onto each sampled negative logit and dominated/collapsed the light tower;
    see the aggregation note below). The Eq.8 similarity runs on the INPUT
    feature vectors ``x`` (paper notation ``sim(x_i, x_z)``: cosine on the
    L2-normalized embedding concat), NOT on the learned light representation.
    The adaptive negative weight (Eq.8) is ``δ · (softmax_j(cos(x_i, x_j))
    |_{j=z} + sigmoid(neg_logit_z))`` where the softmax normalizes over the
    FULL negative pool (paper denominator ``Σ_{j∈D_tr^-}``), not just the
    sampled N; only the sampled negatives' columns enter the Eq.7 sum. This
    weight is a stop-grad measure (computed under ``torch.no_grad()``):
    gradients reach the model only through the Eq.7 logit terms, never
    through the weight itself -- otherwise the ``sigmoid(neg_logit)`` term
    opens a shortcut that pushes negative logits to ``-inf`` (shrinking the
    loss AND the weight simultaneously), which collapsed the light tower in
    smoke training. The Eq.7 InfoNCE math reuses
    :func:`weighted_infonce_loss` (single source of truth).

    ``margin`` (additive logit margin, raw logit units) is forwarded to
    :func:`weighted_infonce_loss`: a pair's
    gradient decays to ~0 once ``p_i - n_iz >= margin + τ·log(Σ_z w_iz)``.
    ``m > 0`` hardens the separation demand; ``m < 0`` is a tolerance that
    relaxes the saturation threshold toward the data's discriminative
    scale (useful when WCL's default threshold ``τ·log(Σw)`` -- ~1.4σ at
    N=32, w̄≈0.5, τ=0.5 -- sits above the honest effect size of the data).

    Sampling is per-positive UNIFORM over the full negative pool. A
    booster-scored hard-negative shortlist (``hard_negative_pool_size``,
    ANSL-style) was implemented 2026-08-24 and REMOVED the same day after
    run-6 falsified it: selecting negatives by the teacher's click score
    targets the "click-like" cluster the positives live in, and the Eq.7
    suppression landed on the whole cluster through the shared light head
    (light click AUC inverted to 0.43 and fell monotonically while the
    booster improved -- no self-correction channel). Score-based hard
    negatives conflict with this architecture (task-logit Eq.7 + hint
    teacher-student pull). See docs/sdcl_rocket_launching_dev.md §10.

    Returns a 0-dim loss tensor: the SUM of per-positive losses (paper
    Eq.7 ``Σ_{i∈D^+}``). Together with per-positive sampling this keeps
    every logit's gradient O(α_w/τ) and independent of the positive count
    P: each positive appears in exactly one row, each negative in
    ``k ≈ N·P/pool`` rows (O(1) for sensible N). A batch MEAN would
    under-scale the per-logit force by P× (~3e-5 at P=300, invisible next
    to BCE's ~0.1). Returns ``0`` when the batch has no positive or no
    negative for this task. The task weight ``β_w^t`` (Eq.7) and total
    weight ``α_w`` (Eq.2) are applied by the caller.
    """
    # ========== 第 0 步：in-batch 按标签划分正负样本 ==========
    # 正样本 = 本 task label==1 的样本；负样本池 = label==0 的样本。
    # （负样本来自同一个 batch，不引入额外采样数据源。）
    pos_mask = label > 0
    neg_mask = label == 0
    n_pos = int(pos_mask.sum().item())
    n_neg_pool = int(neg_mask.sum().item())
    # batch 内没有正样本或没有负样本时，WCL 无法构成对比对，返回 0
    # （该 task 的 WCL 项对本 batch 贡献为 0，不影响其他 loss 项）。
    if n_pos == 0 or n_neg_pool == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    # ========== 第 1 步：为每个正样本独立抽 N 个负样本 ==========
    # N = min(num_negatives, 池大小)。per-positive 采样（2026-08-20）：
    # rand_idx [P, N] 是每个正样本自己的负样本（池内相对位置，有放回），
    # 对应论文 Eq.7 中"每个正样本 i 有自己的负样本集 {z}"。此前是全批
    # 共享 N 个负样本再 expand 成 [P, N]，叠加 sum 聚合会把 ∝P·α/τ 的梯度
    # 集中轰在 N 个（每步随机轮换的）负样本 logit 上（batch 4096、P≈300
    # 时约为 BCE 单 logit 梯度的 100~600 倍），light 塔输出层被随机方向
    # 的强迫项主导，AUC 压到 0.50（smoke 2026-08-20 实证；z-score 修复了
    # 系统性膨胀但对此随机轰击无效）。
    _n = min(num_negatives, n_neg_pool)
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
    neg_pool_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)

    # 每个正样本在全池内独立、均匀抽 N 个负样本（有放回）。
    # （ANSL 硬负例短名单 2026-08-24 实现并于同日移除：run-6 证伪——
    # 按 booster 分数选"难负例"恰好选中正样本所在的 click-like 簇，
    # Eq.7 的压制经共享 light head 落到整簇上，light click AUC 倒挂至
    # 0.43 且随 booster 变好持续恶化，无自愈通道。详见 dev doc §10。）
    neg_local = torch.randint(0, n_neg_pool, (n_pos, _n), device=logits.device)

    # ========== 第 2 步：取出正/负样本的 logit 和输入特征向量 ==========
    # logits 是 light 塔本 task 的原始输出（score，未经 sigmoid）；
    # input_fea 是进模型前的特征向量（embedding concat，论文 Eq.8 的 x），
    # 已由调用方 L2 归一化，点积即余弦相似度。Eq.8 的 softmax 分母要覆盖
    # 整个负样本池，因此取全池特征 neg_pool_fea（而非仅抽中的 N 个）。
    pos_logits = logits[pos_idx]  # [P]
    pos_fea = input_fea[pos_idx]  # [P, d]
    neg_logits = logits[neg_pool_idx[neg_local]]  # [P, N] per-positive negatives
    neg_pool_fea = input_fea[neg_pool_idx]  # [pool, d]

    # ========== 第 3 步：Eq.8 自适应负样本权重 w_iz（stop-grad 度量） ==========
    # w_iz = δ · ( softmax_{j∈负样本池}( cos(x_i, x_j) ) |_{j=z} + sigmoid(n_iz) )
    #   第一项（内容相似度，定义在输入特征向量 x 上）：对正样本 i，在
    #     【整个负样本池】上做 softmax（论文 Eq.8 分母是 Σ_{j∈D_tr^-}，
    #     不是只在抽出的 N 个负样本上归一化），再 gather 出正样本 i 抽中的
    #     负样本 z 的列 —— 与正样本输入特征越相似的负样本（内容空间里的
    #     难负例）权重越大；
    #   第二项（打分难度）：负样本的 light 预测分过 sigmoid ——
    #     模型当前打分越高的负样本（打分空间的难负例）权重越大；
    #   δ 为整体缩放系数。
    # stop-grad（2026-08-19）：w 是纯"度量/重加权系数"，不回传梯度。
    # 不 detach 时 σ(n_iz) 项存在"压低负 logit → w 同步变小"的双重下降
    # 捷径，远端 smoke train 实证 light 被 WCL 梯度主导而塌缩（WCL 731→5.7
    # 而 light AUC→0.5、BCE 涨到 7+）。detach 后梯度只经 Eq.7 的 logit 项
    # 回传；这与 booster 软目标 stop-grad 同理（loss 侧，非前向 detach）。
    with torch.no_grad():
        sim_full = pos_fea @ neg_pool_fea.t()  # [P, pool] cosine (normalized)
        softmax_full = torch.softmax(
            sim_full, dim=1
        )  # [P, pool] softmax over FULL pool
        softmax_sim = torch.gather(softmax_full, 1, neg_local)  # [P, N]
        diff_term = torch.sigmoid(neg_logits)  # [P, N]
        w = delta * (softmax_sim + diff_term)  # [P, N]

    # ========== 第 4 步：Eq.7 加权 InfoNCE（sum 聚合 + margin） ==========
    # loss_i = -log( exp(p_i/τ) / ( exp(p_i/τ) + Σ_z w_iz·exp((n_iz+m)/τ) ) )
    # 数值稳定形式（logsumexp）实现在 weighted_infonce_loss 中：
    #   loss_i = -(p_i/τ) + logsumexp( [p_i/τ, (n_iz+m)/τ + log(w_iz)] )
    # [P, N]：每行 = 一个正样本 vs 它自己抽中的 N 个负样本；对行求和
    # （论文 Σ_i）。per-positive 采样 + sum 下每个 logit 的梯度 =
    # O(α_w/τ)，与 P 无关（正样本只出现在自己那一行，负样本出现
    # k ≈ N·P/pool = O(1) 行）。margin m 平移梯度饱和阈值
    # （p−n ≥ m + τ·logΣw 后该对梯度 →0）：m>0 加严，m<0 容差。
    return weighted_infonce_loss(pos_logits, neg_logits, w, temperature, margin=margin)


class SDCLRocketLaunching(MTLRocketLaunching2):
    """SDCL-style RocketLaunching: ANSL on the light (pre-ranking) net.

    Differences from :class:`MTLRocketLaunching2`:

    1. **Light input detached from the shared bottom** (2026-08-19, v2-style
       isolation): ``_light_forward`` runs on ``share.detach()``, so the
       light-side losses (BCE + hint + WCL) train only ns_gate / light_mlp /
       light heads; the shared bottom (``share_mlp`` / embeddings / booster
       trunk) is trained by the booster alone. Rationale: with joint (no-
       detach) training the light losses dominated ~80% of the total gradient
       and polluted the shared bottom, dragging the booster below baseline
       (smoke train 2026-08-19). The paper's joint training presumes a
       well-trained booster; warmstarting the booster from a ranking
       baseline may later allow restoring joint training.
    2. **Negative Sample Gate Unit** (``ns_gate``, SDCL Eq.6) on the light
       input: ``g = x ⊙ Gate(x)``.
    3. **Weighted Contrastive Loss** (Eq.7 + Eq.8 adaptive weights) on the
       light task logits, per-task opt-in via ``task_wcl_weights``. Weights
       are two-level as in the paper: ``α_w`` (model-level ``wcl_weight``,
       Eq.2) × ``β_w^t`` (``task_wcl_weights.weight``, Eq.7); Eq.7 aggregates
       with the paper's SUM over positives, and each positive draws its OWN
       negatives (per-positive sampling, 2026-08-20 -- sharing N negatives
       across all P positives concentrated an α·P/τ gradient on each sampled
       negative logit, ~300× BCE at batch 4096, which dominated and collapsed
       the light tower; see ``_wcl_for_task``). ``wcl_margin`` shifts the
       gradient-saturation threshold (m<0 = tolerance toward the data's
       effect size). The Eq.8 adaptive weight is a stop-grad measure
       (loss-side, like the booster soft target): without it the
       ``sigmoid(neg_logit)`` term lets WCL shrink its own weight by pushing
       negative logits down, which collapsed the light tower in smoke
       training (see ``_wcl_for_task``). Two WCL variants were tried and
       REMOVED after being falsified by smoke/train runs (details in
       ``_wcl_for_task`` and the dev doc): booster-scored hard-negative
       shortlists (``hard_negative_pool_size``, run-6: light click AUC
       inverted to 0.43) and logit z-scoring (``wcl_standardize_logit``,
       2026-08-20: eval AUC pinned at 0.500).
    4. **Distillation = logit BCE only** (Eq.5; ``HINT_BCE`` default). The
       feature-based similarity loss is dropped (not in the paper).
    5. **Booster-only detach** (``detached_tower_names``, 2026-08-21): towers
       listed there read the booster trunk detached, so their task loss does
       not update the shared bottom -- the bottom is trained by the remaining
       towers only (e.g. click-exclusive = the v3 baseline booster recipe;
       run-4 showed the attached, task-space-weighted CVR diluting the click
       booster by ~0.5pt AUC). Unlike :class:`MTLRocketLaunching2` the light
       side is NOT detached: the listed towers' light heads keep updating
       ``ns_gate`` / ``light_mlp`` (run-4's +6pt light-CVR gain came from
       exactly that light-side multi-task signal). ``light_mlp`` itself is
       optional since 2026-08-21: when omitted, the (gated) share feeds the
       light heads directly.

    The booster (ranking net) is unchanged from v2: ``share -> cross ‖ deep
    (each + LayerNorm, concat) -> per-task (task_mlp + Linear)``, training only.

    The distillation loss keeps a ``.detach()`` on the booster logit *as the
    soft target* (Eq.5 uses the fixed ``R_rank``) -- this is a loss-side
    stop-grad on the teacher, not a forward-path detach.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Skip MTLRocketLaunching2.__init__ (it reads
        # feature_based_distillation, which this proto does not have, and its
        # detached_tower_names detaches BOTH the booster and light sides) and
        # go straight to MultiTaskRank.__init__, then rebuild the (feature-
        # sim-free, light-input-detached) booster + light with the NSGate and
        # WCL config. SDCL's own detached_tower_names is BOOSTER-ONLY (see
        # _booster_detached_towers below).
        MultiTaskRank.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert model_config.WhichOneof("model") == "sdcl_rocket_launching", (
            "invalid model config: %s" % model_config.WhichOneof("model")
        )

        self._task_nums = len(self._task_tower_cfgs)

        # ---- distillation weights (logit BCE hint = Eq.5) ----
        _overrides = {w.tower_name: w for w in self._model_config.task_distill_weights}
        self._hint_loss_weights: Dict[str, float] = {}
        self._hint_loss_types: Dict[str, int] = {}
        _known_towers = {c.tower_name for c in self._task_tower_cfgs}
        for _tower_cfg in self._task_tower_cfgs:
            _tname = _tower_cfg.tower_name
            _ov = _overrides.get(_tname)
            self._hint_loss_weights[_tname] = (
                _ov.hint_loss_weight
                if _ov is not None and _ov.HasField("hint_loss_weight")
                else self._model_config.hint_loss_weight
            )
            self._hint_loss_types[_tname] = (
                _ov.hint_loss_type
                if _ov is not None and _ov.HasField("hint_loss_type")
                else self._model_config.hint_loss_type
            )
        for _w in self._model_config.task_distill_weights:
            if _w.tower_name not in _known_towers:
                logging.warning(
                    "task_distill_weights tower_name %s not found in task_towers,"
                    " this override has no effect.",
                    _w.tower_name,
                )

        # ---- booster-only detach (2026-08-21): towers listed here read the
        # booster trunk DETACHED in predict(), so their task loss does not
        # update the shared bottom (share_mlp / cross / deep / embeddings) --
        # the bottom is trained by the remaining towers only (e.g.
        # click-exclusive, the v3 baseline booster recipe; smoke run-4 showed
        # the attached CVR diluting the click booster by ~0.5pt AUC). The
        # tower's own booster task mlp + linear still learn. Unlike
        # MTLRocketLaunching2 the LIGHT side is NOT detached: the listed
        # towers' light heads keep updating ns_gate / light_mlp (that
        # light-side multi-task signal is what lifted light CVR +6pt over
        # baseline in run-4).
        self._booster_detached_towers = set(self._model_config.detached_tower_names)
        _unknown_detach = self._booster_detached_towers - _known_towers
        if _unknown_detach:
            logging.warning(
                "detached_tower_names %s not found in task_towers,"
                " these names have no effect.",
                sorted(_unknown_detach),
            )

        self._label_by_tower: Dict[str, str] = {
            c.tower_name: c.label_name for c in self._task_tower_cfgs
        }

        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_in = self.embedding_group.group_total_dim(self.group_name)

        # ===== Shared bottom (trained by the booster; light reads it detached) =====
        self.share_mlp = None
        if self._model_config.HasField("share_mlp"):
            self.share_mlp = MLP(
                feature_in, **config_to_kwargs(self._model_config.share_mlp)
            )
        share_dim = self.share_mlp.output_dim() if self.share_mlp else feature_in

        # ===== Booster trunk: cross & deep run in PARALLEL from share, each
        # with its own LayerNorm; outputs concatenated -> task towers. (No PLE,
        # no feature_based_distillation -> task mlps return plain tensors.) =====
        self.booster_cross = None
        self.booster_cross_ln = None
        if self._model_config.HasField("cross"):
            self.booster_cross = CrossV2(
                input_dim=share_dim,
                **config_to_kwargs(self._model_config.cross),
            )
            self.booster_cross_ln = nn.LayerNorm(self.booster_cross.output_dim())

        self.booster_deep = None
        self.booster_deep_ln = None
        if self._model_config.HasField("deep"):
            self.booster_deep = MLP(
                in_features=share_dim,
                **config_to_kwargs(self._model_config.deep),
            )
            self.booster_deep_ln = nn.LayerNorm(self.booster_deep.output_dim())

        trunk_dim = 0
        if self.booster_cross is not None:
            trunk_dim += self.booster_cross.output_dim()
        if self.booster_deep is not None:
            trunk_dim += self.booster_deep.output_dim()
        if trunk_dim == 0:
            trunk_dim = share_dim

        self._extraction_nets = nn.ModuleList()  # no PLE

        # ===== Booster: per-task towers (task mlp + linear) =====
        self.booster_task_mlps = nn.ModuleDict()
        self.booster_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            task_in = trunk_dim
            if task_tower_cfg.HasField("mlp"):
                self.booster_task_mlps[tower_name] = MLP(
                    task_in, **config_to_kwargs(task_tower_cfg.mlp)
                )
                task_in = self.booster_task_mlps[tower_name].output_dim()
            self.booster_task_outputs[tower_name] = nn.Linear(
                task_in, task_tower_cfg.num_class
            )

        # ===== Light (pre-ranking net, training + inference): reads the
        # shared bottom DETACHED (light losses train only light-side params) =====
        # light_mlp is optional since 2026-08-21: when omitted, the (gated)
        # share feeds the light heads directly (share_dim; the NSGate is
        # dim-preserving so the dims line up either way).
        self.light_mlp = None
        if self._model_config.HasField("light_mlp"):
            self.light_mlp = MLP(
                share_dim, **config_to_kwargs(self._model_config.light_mlp)
            )
        light_rep_dim = (
            self.light_mlp.output_dim() if self.light_mlp is not None else share_dim
        )
        # Negative Sample Gate Unit on the light input (Eq.6). hidden_units
        # is repeated (one line per gate layer); empty -> NSGate resolves
        # [input_dim // 4] internally. per_task: true gives each task tower
        # its OWN gate, applied to the TASK TOWER INPUT (PEPNet-style:
        # directly on the detached shared bottom when light_mlp is absent,
        # or after the shared light_mlp when present -- light_mlp runs
        # once, the gates are per task) instead of one shared gate on
        # share. Zero-init keeps Gate = alpha/2 at start for every task.
        self.ns_gate = None
        self.task_ns_gates = None
        if self._model_config.HasField("ns_gate"):
            _ns = self._model_config.ns_gate
            if _ns.per_task:
                self.task_ns_gates = nn.ModuleDict()
                for task_tower_cfg in self._task_tower_cfgs:
                    self.task_ns_gates[task_tower_cfg.tower_name] = NSGate(
                        input_dim=light_rep_dim,
                        hidden_units=list(_ns.hidden_units),
                        alpha=_ns.alpha,
                    )
            else:
                self.ns_gate = NSGate(
                    input_dim=share_dim,
                    hidden_units=list(_ns.hidden_units),
                    alpha=_ns.alpha,
                )
        self._has_light_task_mlp = self._model_config.HasField("light_task_mlp")
        self.light_task_mlps = nn.ModuleDict()
        self.light_task_outputs = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            if self._has_light_task_mlp:
                self.light_task_mlps[tower_name] = MLP(
                    light_rep_dim,
                    **config_to_kwargs(self._model_config.light_task_mlp),
                )
                head_in = self.light_task_mlps[tower_name].output_dim()
            else:
                head_in = light_rep_dim
            self.light_task_outputs[tower_name] = nn.Linear(
                head_in, task_tower_cfg.num_class
            )

        # ===== WCL config (per-task opt-in) =====
        # Two weight levels, as in the paper: α_w (model-level wcl_weight,
        # Eq.2 total WCL weight) × β_w^t (task_wcl_weights.weight, Eq.7
        # per-task weight, default 1.0 when unset -- it no longer falls back
        # to wcl_weight). Effective per-task coefficient = α_w · β_w^t.
        self._wcl_alpha = self._model_config.wcl_weight
        # Additive logit margin for Eq.7 (raw logit units): shifts the
        # gradient-saturation threshold p - n >= m + τ·log(Σw). m > 0
        # hardens the separation demand; m < 0 is a tolerance relaxing it
        # toward the data's discriminative scale.
        # (wcl_standardize_logit removed 2026-08-24: batch z-scoring the
        # logits collapsed light AUC to 0.500 in the 2026-08-20 smokes.)
        self._wcl_margin = self._model_config.wcl_margin
        self._wcl_cfgs: Dict[str, Dict[str, float]] = {}
        _wcl_overrides = {w.tower_name: w for w in self._model_config.task_wcl_weights}
        for task_tower_cfg in self._task_tower_cfgs:
            _tname = task_tower_cfg.tower_name
            _ov = _wcl_overrides.get(_tname)
            if _ov is None or not _ov.enable:
                continue
            self._wcl_cfgs[_tname] = {
                "weight": _ov.weight if _ov.HasField("weight") else 1.0,
                "num_negatives": (
                    _ov.num_negatives
                    if _ov.HasField("num_negatives")
                    else self._model_config.wcl_num_negatives
                ),
                "temperature": (
                    _ov.temperature
                    if _ov.HasField("temperature")
                    else self._model_config.wcl_temperature
                ),
                "delta": (
                    _ov.delta if _ov.HasField("delta") else self._model_config.wcl_delta
                ),
                # (hard_negative_pool_size removed 2026-08-24 -- run-6
                # falsified booster-scored shortlists; sampling is always
                # per-positive uniform over the full pool.)
            }

    def _light_forward(self, share: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the light branch on ``share.detach()`` (v2-style isolation).

        The light branch's losses (BCE + hint + WCL) must NOT update the
        shared bottom (embeddings / share_mlp / booster trunk): in smoke
        training the light-side losses dominated ~80% of the total and their
        distorted gradients polluted the shared bottom, dragging the booster
        below baseline. With the detach, the shared bottom is trained by the
        booster alone (as in :class:`MTLRocketLaunching2`); light-side
        parameters (ns_gate / light_mlp / light heads) still learn normally.

        Returns:
            tower_logits: per-tower raw output tensor (no feature-sim hidden
                features -- feature_based_distillation is dropped).
        """
        share = share.detach()
        # Shared gate path (original Eq.6): one gate on share, then the
        # shared light_mlp (if any), then per-task heads.
        if self.task_ns_gates is None:
            light_in = self.ns_gate(share) if self.ns_gate is not None else share
            light_rep = (
                self.light_mlp(light_in) if self.light_mlp is not None else light_in
            )
            tower_logits: Dict[str, torch.Tensor] = {}
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                if tower_name in self.light_task_mlps:
                    raw = self.light_task_mlps[tower_name](light_rep)
                    tower_logits[tower_name] = self.light_task_outputs[tower_name](raw)
                else:
                    tower_logits[tower_name] = self.light_task_outputs[tower_name](
                        light_rep
                    )
            return tower_logits
        # Per-task gate path: the shared light_mlp (if any) runs ONCE on the
        # detached share; each task tower then applies its OWN gate to its
        # input (gate dims = light_rep_dim) before its task mlp / head.
        light_rep = self.light_mlp(share) if self.light_mlp is not None else share
        tower_logits = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            task_in = self.task_ns_gates[tower_name](light_rep)
            if tower_name in self.light_task_mlps:
                raw = self.light_task_mlps[tower_name](task_in)
                tower_logits[tower_name] = self.light_task_outputs[tower_name](raw)
            else:
                tower_logits[tower_name] = self.light_task_outputs[tower_name](task_in)
        return tower_logits

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        net = grouped_features[self.group_name]
        if self.share_mlp is not None:
            share = self.share_mlp(net)
        else:
            share = net

        # ---- Light branch (training + inference); input detached from the
        # shared bottom (v2-style: light losses do not update share/embeddings)
        light_logits = self._light_forward(share)
        predictions = self._tower_outputs_to_predictions(light_logits, "_light")
        if self.training:
            # Cache the INPUT feature vectors (pre share_mlp) for WCL's Eq.8
            # similarity sim(x_i, x_z) -- paper-defined on the input x, not on
            # the learned light representation (training only; keep eval clean).
            predictions["wcl_input_fea"] = net

        # ---- Booster branch (training only) ----
        if self.training:
            parallel_outputs: List[torch.Tensor] = []
            if self.booster_cross is not None:
                parallel_outputs.append(
                    self.booster_cross_ln(self.booster_cross(share))
                )
            if self.booster_deep is not None:
                parallel_outputs.append(self.booster_deep_ln(self.booster_deep(share)))
            if len(parallel_outputs) > 1:
                trunk = torch.cat(parallel_outputs, dim=-1)
            elif len(parallel_outputs) == 1:
                trunk = parallel_outputs[0]
            else:
                trunk = share

            booster_logits: Dict[str, torch.Tensor] = {}
            for task_tower_cfg in self._task_tower_cfgs:
                tower_name = task_tower_cfg.tower_name
                rep = trunk
                # booster-only detach: the listed tower's task loss must not
                # reach the shared bottom; its own task mlp + linear still
                # learn (grad is cut BEFORE the task mlp, not after).
                if tower_name in self._booster_detached_towers:
                    rep = rep.detach()
                if tower_name in self.booster_task_mlps:
                    rep = self.booster_task_mlps[tower_name](rep)
                booster_logits[tower_name] = self.booster_task_outputs[tower_name](rep)
            predictions.update(
                self._tower_outputs_to_predictions(booster_logits, "_booster")
            )
        return predictions

    def init_loss(self) -> None:
        """Initialize loss modules: BCE task losses + HintLoss (inherited).

        WCL adds no learnable parameters -- the Eq.7 InfoNCE math is a pure
        function (:func:`weighted_infonce_loss`) called from the
        ``@torch.fx.wrap`` leaf :func:`_wcl_for_task`, so nothing to
        instantiate here beyond the inherited BCE + HintLoss modules.
        """
        super().init_loss()  # MTLRocketLaunching.init_loss: BCE + hint modules

    def _distillation_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        loss_weights: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Per-tower logit BCE hint distillation (Eq.5). No feature similarity."""
        losses: Dict[str, torch.Tensor] = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            logits_light = predictions[f"logits_{tower_name}_light"]
            logits_booster = predictions[f"logits_{tower_name}_booster"].detach()
            batch_hint = self.hint_loss_modules[tower_name](
                logits_light, logits_booster
            )
            lw = loss_weights[tower_name]
            if lw is not None:
                hint = torch.mean(batch_hint * lw)
            else:
                hint = batch_hint
            losses[f"hint_l2_loss_{tower_name}"] = (
                hint * self._hint_loss_weights[tower_name]
            )
        return losses

    def _wcl_losses(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Weighted Contrastive Loss (Eq.7 + Eq.8) on the light task logits.

        Per enabled task (opt-in via task_wcl_weights): in-batch positives are
        ``label == 1`` and the negative pool is ``label == 0``; each positive
        draws its own N negatives (per-positive sampling). The adaptive
        negative weight (Eq.8) is ``δ · (softmax over the FULL negative pool
        of cos(x_i, x_j), evaluated at the sampled negatives +
        sigmoid(neg_logit_z))`` with the similarity on the INPUT feature
        vectors x (paper Eq.8), not on the learned light representation.

        Weights: the per-task weight ``β_w^t`` (``task_wcl_weights.weight``,
        default 1.0) and the total weight ``α_w`` (model-level ``wcl_weight``)
        multiply the per-positive SUMMED loss (paper Eq.7 ``Σ_i``; effective
        per-task coefficient = ``α_w · β_w^t``). With per-positive sampling
        the sum keeps every logit's gradient O(α_w/τ) regardless of the
        positive count P -- the 2026-08-20 light collapses traced back to
        sharing N negatives across all P positives UNDER the sum, which
        concentrated an ``α·P/τ`` gradient on each sampled negative logit
        (~300× BCE at batch 4096). ``wcl_margin`` shifts the gradient
        saturation threshold (``p - n >= m + τ·log(Σw)``; m<0 = tolerance
        toward the data's effect size).

        The Eq.8 adaptive weight is a stop-grad measure (computed under
        ``torch.no_grad()`` inside ``_wcl_for_task``): gradients flow only
        through the Eq.7 logit terms.

        (Booster-scored hard-negative sampling and logit z-scoring were
        both implemented here once and removed after being falsified --
        run-6 and the 2026-08-20 smokes respectively; sampling is now
        always per-positive uniform over the full pool, and raw logits
        feed Eq.7 directly. See ``_wcl_for_task`` / dev doc §10.)
        """
        losses: Dict[str, torch.Tensor] = {}
        if "wcl_input_fea" not in predictions:
            return losses
        # Paper Eq.8: sim(x_i, x_z) on the input feature vectors, L2-normalized
        # so the dot product inside _wcl_for_task is cosine similarity.
        fea = F.normalize(predictions["wcl_input_fea"], p=2, dim=1)
        for _tname, _cfg in self._wcl_cfgs.items():
            label = (
                batch.labels[self._label_by_tower[_tname]].to(torch.float32).reshape(-1)
            )
            logits = predictions[f"logits_{_tname}_light"].reshape(-1)
            # The data-dependent sampling + Eq.8 weighting lives behind the
            # @torch.fx.wrap leaf _wcl_for_task so this loss() path stays
            # torch.fx-traceable for TorchRec's train pipeline.
            wcl = _wcl_for_task(
                fea,
                logits,
                label,
                int(_cfg["num_negatives"]),
                _cfg["temperature"],
                _cfg["delta"],
                margin=self._wcl_margin,
            )
            # Eq.7 per-task weight β_w^t × Eq.2 total weight α_w on the
            # per-positive SUMMED loss (paper Σ_i; per-positive sampling
            # keeps the per-logit gradient O(α_w/τ), P-independent -- see
            # _wcl_for_task). Summing the loss-dict entries in TrainWrapper
            # then yields α_w · Σ_t β_w^t · Σ_i ℓ_i^t.
            losses[f"weighted_infonce_{_tname}_light"] = (
                wcl * self._wcl_alpha * _cfg["weight"]
            )
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss: rank BCE + distill (Eq.5) + WCL (Eq.7+8)."""
        losses = super().loss(predictions, batch)
        # super().loss (MTLRocketLaunching.loss) returns an OrderedDict of
        # rank + _loss_collection + _distillation_losses (my override).
        # Append WCL (training only); keep an OrderedDict for stable ordering.
        if self.training:
            wcl = self._wcl_losses(predictions, batch)
            if wcl:
                if not isinstance(losses, OrderedDict):
                    losses = OrderedDict(losses)
                losses.update(wcl)
        return losses
