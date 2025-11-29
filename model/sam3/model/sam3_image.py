# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model_misc import SAM3Output

from .vl_combiner import SAM3VLBackbone

from .data_misc import BatchedDatapoint

from .act_ckpt_utils import activation_ckpt_wrapper

from .box_ops import box_cxcywh_to_xyxy

from .geometry_encoders import Prompt
from .model_misc import inverse_sigmoid


def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(torch.nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multigpu=False,
        gather_backbone_out=False,
        async_all_gather=False,
        o2m_mlp_ratio=2.0,
        separate_scorer_for_instance=False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.input_geometry_encoder = input_geometry_encoder
        self.segmentation_head = segmentation_head
        self.num_feature_levels = num_feature_levels
        self.o2m_mask_predict = o2m_mask_predict
        self.dot_prod_scoring = dot_prod_scoring
        self.use_instance_query = use_instance_query
        self.inst_interactive_predictor = inst_interactive_predictor
        self.num_interactive_steps_val = num_interactive_steps_val

        # multi-GPU support
        self.multigpu = multigpu
        self.gather_backbone_out = gather_backbone_out
        self.async_all_gather = async_all_gather
        self.world_size = torch.distributed.get_world_size() if multigpu else 1
        self.rank = torch.distributed.get_rank() if multigpu else 0

        # whether to use a separate scorer for instance queries
        self.separate_scorer_for_instance = separate_scorer_for_instance

        # instance query related
        self.num_queries = transformer.decoder.num_queries
        self.num_o2m_queries = transformer.decoder.num_o2m_queries
        self.num_total_queries = self.num_queries + self.num_o2m_queries
        self.num_dac_queries = self.num_total_queries
        self.dac = transformer.decoder.dac

        # build box head
        hidden_dim = transformer.d_model
        self.bbox_embed = torch.nn.Linear(hidden_dim, 4)
        self.bbox_embed.weight.data.normal_(mean=0.0, std=0.001)
        self.bbox_embed.bias.data.zero_()

        # build mask head
        self.mask_embed = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mask_embed.weight.data.normal_(mean=0.0, std=0.02)
        self.mask_embed.bias.data.zero_()

        # build presence token head
        self.presence_token = transformer.decoder.presence_token
        if self.presence_token is not None:
            self.presence_token_head = torch.nn.Linear(hidden_dim, 1)
            self.presence_token_head.weight.data.normal_(mean=0.0, std=0.001)
            self.presence_token_head.bias.data.zero_()
            self.presence_token_out_norm = torch.nn.LayerNorm(hidden_dim)
            self.clamp_presence_logits = True
            self.clamp_presence_logit_max_val = 10.0

        # build o2m mask prediction head
        if self.o2m_mask_predict:
            self.o2m_mask_embed = torch.nn.Linear(hidden_dim, hidden_dim)
            self.o2m_mask_embed.weight.data.normal_(mean=0.0, std=0.02)
            self.o2m_mask_embed.bias.data.zero_()
            self.o2m_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, int(hidden_dim * o2m_mlp_ratio)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(int(hidden_dim * o2m_mlp_ratio), hidden_dim),
            )
            self.o2m_mlp.weight.data.normal_(mean=0.0, std=0.02)
            self.o2m_mlp.bias.data.zero_()

        # instance query related
        if self.use_instance_query:
            self.instance_norm = torch.nn.LayerNorm(hidden_dim)

        # box refinement
        self.box_refine = transformer.decoder.box_refine
        self.use_normed_output_consistently = True

        # add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        self.no_mem_embed = torch.nn.Embedding(1, hidden_dim)
        self.no_mem_embed.weight.data.normal_(mean=0.0, std=0.02)

        # set to eval mode by default
        self.eval()

    def forward_grounding(
        self,
        backbone_out,
        find_input,
        geometric_prompt: Prompt,
        find_target=None,
        **kwargs,
    ):
        """
        Forward pass for grounding.
        """
        # Add no_mem_embed, which is added to the lowest res feat. map during training on videos
        if "memory" in backbone_out and backbone_out["memory"] is not None:
            backbone_out["memory"] = backbone_out["memory"] + self.no_mem_embed.weight[0]

        # Process geometric prompts
        geometric_prompt_embed = self.input_geometry_encoder(geometric_prompt)

        # Run transformer
        transformer_out = self.transformer(
            backbone_out=backbone_out,
            geometric_prompt_embed=geometric_prompt_embed,
            find_input=find_input,
        )

        # Process transformer outputs
        hs = transformer_out["hs"]
        memory = transformer_out["memory"]
        memory_pos = transformer_out["memory_pos"]
        spatial_shapes = transformer_out["spatial_shapes"]
        level_start_index = transformer_out["level_start_index"]
        valid_ratios = transformer_out["valid_ratios"]
        reference_points = transformer_out["reference_points"]
        prev_encoder_out = transformer_out["prev_encoder_out"]

        # Prepare outputs
        outputs = {}
        outputs["prev_encoder_out"] = prev_encoder_out

        # Process box predictions
        outputs["pred_logits"] = torch.zeros(1, self.num_queries, 1, device=hs.device)
        outputs["pred_boxes"] = torch.zeros(1, self.num_queries, 4, device=hs.device)
        outputs["pred_boxes_xyxy"] = box_cxcywh_to_xyxy(outputs["pred_boxes"])

        # Process mask predictions
        if self.segmentation_head is not None:
            outputs["pred_masks"] = torch.zeros(1, self.num_queries, spatial_shapes[0, 0], spatial_shapes[0, 1], device=hs.device)

        return outputs

    def _get_dummy_prompt(self):
        """Get dummy geometric prompt."""
        return Prompt(
            points=None,
            boxes=None,
            masks=None,
            labels=None,
            points_mask=None,
            boxes_mask=None,
            masks_mask=None,
        )
