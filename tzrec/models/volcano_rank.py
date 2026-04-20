# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.cdot import CompressedDOT
from tzrec.modules.lhuc import LHUCEPGate, LHUCPPNet
from tzrec.modules.mlp import MLP
from tzrec.modules.sequence import DINEncoder
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.rank_model_pb2 import TowerConfig
from tzrec.utils.config_util import config_to_kwargs


class VolcanoRank(RankModel):
    """VolcanoRank: CTR ranking model based on Volcano Engine architecture.

    Three-layer configuration:
        1. feature_configs: input feature definitions (unchanged).
        2. tower_feature_configs: per-tower feature assignment + dim_size.
        3. volcano_rank: model architecture (MLP, CDOT, etc.).

    Each tower (deep, cdot, bias, lhuc) declares its own feature subset
    and dim_size. The framework automatically computes embedding_dim per
    feature as the sum of dim_sizes across all towers that use it, and
    builds split ranges at forward time.

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
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.init_input()

        # --- Parse tower configs ---
        self._towers: Dict[str, TowerConfig] = {}
        for tc in self._base_model_config.tower_feature_configs:
            self._towers[tc.tower_name] = tc

        group_names = self.embedding_group.group_names()

        # --- Classify groups: SEQUENCE vs DEEP ---
        self._sequence_group_names: List[str] = []
        self._deep_group_name: Optional[str] = None

        for gn in group_names:
            gtype = self.embedding_group.group_type(gn)
            if gtype in (model_pb2.SEQUENCE, model_pb2.JAGGED_SEQUENCE):
                self._sequence_group_names.append(gn)
            elif gtype == model_pb2.DEEP:
                if self._deep_group_name is None:
                    self._deep_group_name = gn

        # --- Build per-tower split info ---
        # For each tower, compute split ranges within the shared deep group.
        deep_tc = self._towers.get("deep")
        cdot_tc = self._towers.get("cdot")
        bias_tc = self._towers.get("bias")
        lhuc_tc = self._towers.get("lhuc")

        self._deep_dim = deep_tc.dim_size if deep_tc else 0
        self._cdot_dim = cdot_tc.dim_size if cdot_tc else 0
        self._bias_dim = bias_tc.dim_size if bias_tc else 0
        self._lhuc_dim = lhuc_tc.dim_size if lhuc_tc else 0

        # Get feature dims from the shared DEEP group (all non-seq features
        # live here, with embedding_dim = sum of dim_sizes of towers using them).
        if self._deep_group_name is not None:
            self._deep_feat_dims = self.embedding_group.group_feature_dims(
                self._deep_group_name
            )
        else:
            self._deep_feat_dims = {}

        # Build per-tower feature sets and split ranges.
        # Each feature in deep_feat_dims has embedding_dim = total_emb_dim.
        # For a given tower, we need to know which features it uses and
        # extract its dim_size-sized slice.
        self._deep_tc_features = set(deep_tc.feature_names) if deep_tc else set()
        self._cdot_tc_features = set(cdot_tc.feature_names) if cdot_tc else set()
        self._bias_tc_features = set(bias_tc.feature_names) if bias_tc else set()
        self._lhuc_tc_features = set(lhuc_tc.feature_names) if lhuc_tc else set()

        # For each feature, compute which towers use it to determine offsets.
        # Tower order: deep, cdot, bias, lhuc (fixed by dim_size config).
        self._tower_order = []
        for name in ["deep", "cdot", "bias", "lhuc"]:
            if name in self._towers:
                self._tower_order.append(name)

        # Build split ranges per tower: for each feature the tower uses,
        # extract [tower_offset : tower_offset + dim_size] from the feature's
        # embedding_dim-sized slice.
        self._tower_split_ranges: Dict[str, List[Tuple[int, int]]] = {}
        for tower_name in self._tower_order:
            tc = self._towers[tower_name]
            tower_feat_set = set(tc.feature_names)
            tower_dim = tc.dim_size
            # Compute offset of this tower within the shared embedding
            tower_offset = sum(
                self._towers[t].dim_size
                for t in self._tower_order[: self._tower_order.index(tower_name)]
            )
            ranges = []
            offset = 0
            for fname, dim in self._deep_feat_dims.items():
                if fname in tower_feat_set:
                    ranges.append(
                        (offset + tower_offset, offset + tower_offset + tower_dim)
                    )
                offset += dim
            self._tower_split_ranges[tower_name] = ranges

        # Number of non-vector features for each tower (used for CDOT slots, etc.)
        self._tower_vec_projections: Dict[str, Dict[str, int]] = {}
        for tower_name in self._tower_order:
            tc = self._towers[tower_name]
            vec_map = {}
            for vp in tc.vector_projections:
                vec_map[vp.feature_name] = vp.target_dim
            self._tower_vec_projections[tower_name] = vec_map

        self._tower_nonvec_count: Dict[str, int] = {}
        for tower_name in self._tower_order:
            tc = self._towers[tower_name]
            nonvec = sum(
                1
                for fn in tc.feature_names
                if fn not in self._tower_vec_projections[tower_name]
            )
            self._tower_nonvec_count[tower_name] = nonvec

        # --- Vector projection MLPs per tower ---
        self._vec_mlps = nn.ModuleDict()
        for tower_name in self._tower_order:
            vec_map = self._tower_vec_projections[tower_name]
            if vec_map:
                mlps = {}
                for fname, target_dim in vec_map.items():
                    value_dim = self._deep_feat_dims.get(fname, 0)
                    mlps[fname] = MLP(
                        in_features=value_dim,
                        hidden_units=[target_dim],
                        activation="nn.Tanh",
                    )
                self._vec_mlps[tower_name] = nn.ModuleDict(mlps)

        # --- Sequence split ranges ---
        # Sequence features shared between towers using them.
        # emb_dim per seq feature = sum of dim_sizes of towers with that seq_group.
        self._seq_tower_split: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
        for tower_name in self._tower_order:
            tc = self._towers[tower_name]
            if not tc.sequence_groups:
                continue
            tower_dim = tc.dim_size
            tower_offset = sum(
                self._towers[t].dim_size
                for t in self._tower_order[: self._tower_order.index(tower_name)]
                if self._towers[t].sequence_groups
                and any(
                    sg in self._towers[t].sequence_groups
                    for sg in tc.sequence_groups
                )
            )
            # Build query and sequence split ranges for this tower
            for seq_gn in tc.sequence_groups:
                query_dims = self.embedding_group.group_dims(f"{seq_gn}.query")
                seq_dims = self.embedding_group.group_dims(f"{seq_gn}.sequence")
                q_ranges = []
                off = 0
                for d in query_dims:
                    q_ranges.append(
                        (off + tower_offset, off + tower_offset + tower_dim)
                    )
                    off += d
                s_ranges = []
                off = 0
                for d in seq_dims:
                    s_ranges.append(
                        (off + tower_offset, off + tower_offset + tower_dim)
                    )
                    off += d
                if seq_gn not in self._seq_tower_split:
                    self._seq_tower_split[seq_gn] = {}
                self._seq_tower_split[seq_gn][tower_name] = {
                    "query": q_ranges,
                    "sequence": s_ranges,
                }

        # --- DIN encoders per tower ---
        max_seq_len = (
            int(self._model_config.max_seq_length)
            if self._model_config.HasField("max_seq_length")
            else 0
        )

        self._din_encoders = nn.ModuleDict()
        self._din_mlps = nn.ModuleDict()
        self._din_lns = nn.ModuleDict()
        self._din_output_dims: Dict[str, int] = {}

        for tower_name in self._tower_order:
            tc = self._towers[tower_name]
            if not tc.sequence_groups or not tc.HasField("din_encoder"):
                continue
            attn_kwargs = config_to_kwargs(tc.din_encoder)
            encoders = nn.ModuleList()
            total_din_dim = 0
            for seq_gn in tc.sequence_groups:
                seq_dim = self.embedding_group.group_total_dim(f"{seq_gn}.sequence")
                query_dim = self.embedding_group.group_total_dim(f"{seq_gn}.query")
                enc = DINEncoder(
                    sequence_dim=seq_dim,
                    query_dim=query_dim,
                    input=seq_gn,
                    attn_mlp=attn_kwargs,
                    max_seq_length=max_seq_len,
                )
                encoders.append(enc)
                total_din_dim += enc.output_dim()
            self._din_encoders[tower_name] = encoders

            din_out_dim = total_din_dim
            if tc.HasField("din"):
                self._din_mlps[tower_name] = MLP(
                    in_features=total_din_dim,
                    **config_to_kwargs(tc.din),
                )
                din_out_dim = self._din_mlps[tower_name].output_dim()
            self._din_lns[tower_name] = nn.LayerNorm(din_out_dim)
            self._din_output_dims[tower_name] = din_out_dim

        # --- Deep branch ---
        deep_input_dim = (
            self._deep_dim * self._tower_nonvec_count.get("deep", 0)
            + sum(
                vp.target_dim for vp in (deep_tc.vector_projections if deep_tc else [])
            )
            + self._din_output_dims.get("deep", 0)
        )
        self.deep = MLP(
            in_features=deep_input_dim,
            **config_to_kwargs(self._model_config.deep),
        )
        self.deep_ln = nn.LayerNorm(self.deep.output_dim())

        # --- CDOT branch ---
        cdot_cfg = self._model_config.cdot
        num_slots = self._tower_nonvec_count.get("cdot", 0)
        slot_dim = self._cdot_dim
        cdot_out_per_slot = cdot_cfg.output_dim
        cdot_mid_dim = cdot_cfg.mid_dim if cdot_cfg.mid_dim > 0 else 32
        compress_hu = list(cdot_cfg.compress_hidden_units) or [512, 256]
        self.cdot = CompressedDOT(
            num_slots=num_slots,
            slot_dim=slot_dim,
            output_dim=cdot_out_per_slot,
            mid_dim=cdot_mid_dim,
            compress_hidden_units=compress_hu,
        )
        cdot_allint_dim = num_slots * cdot_out_per_slot
        cdot_mid_total = slot_dim * cdot_out_per_slot
        self.cdot_out_ln = nn.LayerNorm(cdot_allint_dim)
        self.cdot_mid_ln = nn.LayerNorm(cdot_mid_total)

        # --- Bias branch ---
        bias_out_dim = 0
        if self._model_config.HasField("bias") and bias_tc:
            bias_input_dim = self._bias_dim * self._tower_nonvec_count.get("bias", 0)
            self._bias_mlp = MLP(
                in_features=bias_input_dim,
                **config_to_kwargs(self._model_config.bias),
            )
            self._bias_ln = nn.LayerNorm(self._bias_mlp.output_dim())
            bias_out_dim = self._bias_mlp.output_dim()

        # --- Concat dim ---
        cdot_total = (
            cdot_allint_dim
            + cdot_mid_total
            + self._din_output_dims.get("cdot", 0)
            + sum(
                vp.target_dim for vp in (cdot_tc.vector_projections if cdot_tc else [])
            )
        )
        final_input_dim = self.deep.output_dim() + cdot_total + bias_out_dim

        # --- LHUC or standard final MLP ---
        self._use_lhuc = self._model_config.use_lhuc and lhuc_tc is not None
        if self._use_lhuc:
            lhuc_input_dim = self._lhuc_dim * self._tower_nonvec_count.get("lhuc", 0)
            lhuc_gate_hu = (
                list(self._model_config.lhuc_gate.hidden_units)
                if self._model_config.HasField("lhuc_gate")
                else [256]
            )
            self._lhuc_ep_gate = LHUCEPGate(
                input_dim=final_input_dim,
                gate_input_dim=lhuc_input_dim,
                hidden_units=lhuc_gate_hu,
            )
            final_hu = (
                list(self._model_config.final.hidden_units)
                if self._model_config.HasField("final")
                else [512, 256, 128, 1]
            )
            self._lhuc_pp_net = LHUCPPNet(
                input_dim=final_input_dim,
                gate_input_dim=lhuc_input_dim,
                hidden_units=final_hu,
                scale_last=False,
            )
            self.output_mlp = nn.Linear(
                self._lhuc_pp_net.output_dim(), self._num_class, bias=False
            )
        else:
            self._lhuc_ep_gate = None
            self._lhuc_pp_net = None
            if self._model_config.HasField("final"):
                self.final = MLP(
                    in_features=final_input_dim,
                    **config_to_kwargs(self._model_config.final),
                )
                self.output_mlp = nn.Linear(
                    self.final.output_dim(), self._num_class, bias=False
                )
            else:
                self.final = None
                self.output_mlp = nn.Linear(
                    final_input_dim, self._num_class, bias=False
                )

    @staticmethod
    def _split_tensor(
        tensor: torch.Tensor, split_ranges: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Slice specific dimension ranges from a 2-D tensor."""
        if not split_ranges:
            return tensor
        return torch.cat([tensor[:, s:e] for s, e in split_ranges], dim=-1)

    @staticmethod
    def _split_tensor_3d(
        tensor: torch.Tensor, split_ranges: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Slice specific dimension ranges from a 3-D tensor."""
        if not split_ranges:
            return tensor
        return torch.cat([tensor[:, :, s:e] for s, e in split_ranges], dim=-1)

    def _extract_tower_features(
        self,
        tower_name: str,
        main_emb: torch.Tensor,
        feature_dict: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], int]:
        """Extract and project features for a tower.

        Returns:
            (parts, din_dim): list of feature tensors and DIN output dim.
        """
        tc = self._towers[tower_name]
        ranges = self._tower_split_ranges.get(tower_name, [])
        parts = []
        if ranges:
            parts.append(self._split_tensor(main_emb, ranges))

        # Vector projections
        vec_mlps = (
            self._vec_mlps[tower_name]
            if tower_name in self._vec_mlps
            else {}
        )
        if vec_mlps:
            offset = 0
            for fname, dim in self._deep_feat_dims.items():
                if fname in vec_mlps:
                    vec_slice = main_emb[:, offset : offset + dim]
                    parts.append(vec_mlps[fname](vec_slice))
                offset += dim

        # DIN
        din_dim = self._din_output_dims.get(tower_name, 0)
        if din_dim > 0 and tower_name in self._din_encoders:
            din_parts = []
            for i, seq_gn in enumerate(tc.sequence_groups):
                query = feature_dict[f"{seq_gn}.query"]
                sequence = feature_dict[f"{seq_gn}.sequence"]
                seq_split = self._seq_tower_split.get(seq_gn, {}).get(tower_name)
                if seq_split:
                    query = self._split_tensor(query, seq_split["query"])
                    sequence = self._split_tensor_3d(sequence, seq_split["sequence"])
                seq_embedded = {
                    f"{seq_gn}.query": query,
                    f"{seq_gn}.sequence": sequence,
                    f"{seq_gn}.sequence_length": feature_dict[
                        f"{seq_gn}.sequence_length"
                    ],
                }
                din_parts.append(self._din_encoders[tower_name][i](seq_embedded))
            din_cat = torch.cat(din_parts, dim=-1)
            if tower_name in self._din_mlps:
                din_out = self._din_lns[tower_name](self._din_mlps[tower_name](din_cat))
            else:
                din_out = self._din_lns[tower_name](din_cat)
            parts.append(din_out)

        return parts, din_dim

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Returns:
            predictions (dict): a dict of predicted result.
        """
        feature_dict = self.build_input(batch)
        main_emb = feature_dict[self._deep_group_name]

        # --- Deep branch ---
        deep_parts, _ = self._extract_tower_features("deep", main_emb, feature_dict)
        deep_out = self.deep_ln(self.deep(torch.cat(deep_parts, dim=-1)))

        # --- CDOT branch ---
        cdot_parts, cdot_din_dim = self._extract_tower_features(
            "cdot", main_emb, feature_dict
        )
        # First part is nonvec features (for CDOT slot input)
        cdot_input = cdot_parts[0]
        allint_out, allint_mid_out = self.cdot(cdot_input)
        cdot_out_parts = [
            self.cdot_out_ln(allint_out),
            self.cdot_mid_ln(allint_mid_out),
        ]
        # Add remaining parts (vector projections, DIN output)
        for p in cdot_parts[1:]:
            cdot_out_parts.append(p)
        cdot_out = torch.cat(cdot_out_parts, dim=-1)

        # --- Bias branch ---
        bias_out = None
        if hasattr(self, "_bias_mlp"):
            bias_parts, _ = self._extract_tower_features("bias", main_emb, feature_dict)
            bias_out = self._bias_ln(self._bias_mlp(torch.cat(bias_parts, dim=-1)))

        # --- Concat all branches ---
        branch_outputs = [deep_out, cdot_out]
        if bias_out is not None:
            branch_outputs.append(bias_out)
        net = torch.cat(branch_outputs, dim=-1)

        # --- LHUC or standard final ---
        if self._use_lhuc:
            lhuc_parts, _ = self._extract_tower_features("lhuc", main_emb, feature_dict)
            lhuc_features = torch.cat(lhuc_parts, dim=-1)
            net = self._lhuc_ep_gate(net, lhuc_features)
            logits = self.output_mlp(self._lhuc_pp_net(net, lhuc_features))
        else:
            logits = self.output_mlp(self.final(net) if self.final is not None else net)

        return self._output_to_prediction(logits)
