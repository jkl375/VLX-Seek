import torch
import torch.nn as nn
from typing import List, Optional, Union
import torch.nn.functional as F
import math
import copy
from .simple_fpn import SimpleFP

from .roi_align import roi_align



def gen_sineembed_for_position(pos_tensor, dim_of_pos_feats):
    """Generate sine position embedding from a position tensor.

    Args:
        pos_tensor (torch.Tensor): shape: [batch_size, N, 4]. the last dimension is [cx, cy, w, h] in
            normalized coordinates in range [0, 1].
        out_dim (int): the output dimension of the position embedding.

    Returns:
        pos (torch.Tensor): shape: [batch_size, N, out_dim].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        dim_of_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = 10000 ** (2 * (dim_t // 2) / dim_of_pos_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class HybridFineGrainedRegionEncoder(nn.Module):
    """HFRE: extract fine-grained object prompts from hybrid visual features.

    Args:
        output_size: Spatial resolution used by RoI Align before pooling.
        aux_feature_dims: Channel dimensions of auxiliary visual features.
        spatial_scale: Auxiliary feature-map scale relative to the input image.
        add_pos_embedding: Add sine embeddings derived from normalized boxes.
        pos_embedding_dim: Output dimension of positional embeddings.
        use_vision_tower_object_feature: Include features from the main vision tower.
        object_feature_combination: Strategy for merging auxiliary and tower features.
    """

    def __init__(
        self,
        output_size: int = None,
        aux_feature_dims: List[int] = None,
        spatial_scale: float = None,
        add_pos_embedding: bool = False,
        pos_embedding_dim: int = 1024,
        use_vision_tower_object_feature: bool = False,
        vision_tower_object_feature_dim: int = 5120,
        vision_tower_spatial_scale: float = 1/14,
        object_feature_combination: str = 'mean',
        use_vt_object_feature_only: bool = False,
        use_simpleFPN_for_vt: bool = False,
        use_separate_mlp_for_object: bool = False,
        obj_pooling_type: str = 'mean',  # 'mean', 'max',
        use_multi_scale_roi_align: bool = False,
        apply_object_layer_norm: bool = False,
        roi_algined: bool = False,
        use_simpleFPN_for_vt_aux: bool = False,
        simpleFPN_out_channels_for_vt: int = 512,
    ):
        super().__init__()
        self.output_size = output_size
        self.aux_feature_dims = aux_feature_dims
        self.spatial_scale = spatial_scale
        self.add_pos_embedding = add_pos_embedding
        self.pos_embedding_dim = pos_embedding_dim
        self.use_vision_tower_object_feature = use_vision_tower_object_feature
        self.vision_tower_object_feature_dim = vision_tower_object_feature_dim
        self.vision_tower_spatial_scale = vision_tower_spatial_scale
        self.object_feature_combination = object_feature_combination
        self.use_vt_object_feature_only = use_vt_object_feature_only
        self.use_simpleFPN_for_vt = use_simpleFPN_for_vt
        self.use_separate_mlp_for_object = use_separate_mlp_for_object
        self.obj_pooling_type = obj_pooling_type
        self.use_multi_scale_roi_align = use_multi_scale_roi_align
        self.apply_object_layer_norm = apply_object_layer_norm
        self.roi_algined = roi_algined
        self.aux_feature_dims = aux_feature_dims
        self.use_simpleFPN_for_vt_aux = use_simpleFPN_for_vt_aux
        
        if self.use_vision_tower_object_feature and self.object_feature_combination in ['mean', 'mean_aux_pos']:
            self.vision_tower_object_feature_projector = nn.Sequential(
                nn.Linear(vision_tower_object_feature_dim, pos_embedding_dim),
                nn.GELU(),
                nn.Linear(pos_embedding_dim, pos_embedding_dim)
            )
        
        if self.use_simpleFPN_for_vt:
            self.simple_fpn_dim = self.vision_tower_object_feature_dim
            self.simple_fpn_scale_factors = [8.0, 4.0, 2.0, 1.0]
            self.simple_fpn_stride = 32
            self.simple_fpn_out_channels = simpleFPN_out_channels_for_vt

            self.simple_fpn = SimpleFP(out_channels=self.simple_fpn_out_channels, norm="LN", square_pad=0, dim=self.simple_fpn_dim, stride=self.simple_fpn_stride, scale_factors=self.simple_fpn_scale_factors)
        
        if self.use_simpleFPN_for_vt_aux:
            # Auxiliary features originate from the C-RADIOv4 encoder (1152 channels).
            self.simple_fpn_dim_aux = self.aux_feature_dims[0]
            self.simple_fpn_scale_factors_aux = [4.0, 2.0, 1.0, 0.5]
            self.simple_fpn_stride_aux = 16
            self.simple_fpn_out_channels_aux = 1024
            self.simple_fpn_aux = SimpleFP(out_channels=self.simple_fpn_out_channels_aux, norm="LN", square_pad=0, dim=self.simple_fpn_dim_aux, stride=self.simple_fpn_stride_aux, scale_factors=self.simple_fpn_scale_factors_aux)
        
        if self.use_vision_tower_object_feature and self.use_separate_mlp_for_object:
            self.object_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )
            self.aux_object_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )

    def _apply_pooling(self, level_roi_feat):
        """Pool each RoI feature map according to the configured strategy."""
        if self.obj_pooling_type == 'max':
            return F.adaptive_max_pool2d(level_roi_feat, (1, 1)).squeeze(-1).squeeze(-1)
        else:  # Default to global average pooling.
            return level_roi_feat.mean(dim=(2, 3))

    def extract_vt_object_feature(self, multi_level_features, boxes: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Extract pooled object features from main vision-tower feature maps."""
        if self.use_simpleFPN_for_vt:
            multi_level_features = self.simple_fpn(multi_level_features)

            roi_features_per_level = []
            # Convert each FPN level's stride to the RoI Align spatial scale.
            feature_strides = [ self.simple_fpn_stride / self.simple_fpn_scale_factors[level_idx] for level_idx in range(len(self.simple_fpn_scale_factors))]
            
            for level_idx, level_feature in enumerate(multi_level_features):
                current_spatial_scale = 1.0 / feature_strides[level_idx]
                
                level_roi_feat = roi_align(
                    level_feature.float(),
                    boxes,
                    output_size=self.output_size,
                    spatial_scale=current_spatial_scale,
                    aligned=self.roi_algined
                )
                
                level_roi_feat = level_roi_feat.mean(dim=(2, 3))
                
                roi_features_per_level.append(level_roi_feat)
            
            out_box_feat = torch.cat(roi_features_per_level, dim=1).unsqueeze(0)
        else:
            concat_multi_level_feature = []
            concat_multi_level_feature = torch.cat(multi_level_features, dim=1)

            out_box_feat = roi_align(
                concat_multi_level_feature.float(),
                boxes,
                output_size=self.output_size,
                spatial_scale=self.vision_tower_spatial_scale,
                aligned=self.roi_algined
            )

            # Pool spatial RoI features and restore the expected batch dimension.
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )
            if self.object_feature_combination in ['mean', 'mean_aux_pos']:
                out_box_feat = self.vision_tower_object_feature_projector(out_box_feat.to(concat_multi_level_feature.dtype)).float()
        return out_box_feat
    
    def __call__(
        self,
        multi_level_features: List[torch.Tensor],
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        vt_multi_level_features = None,
        vt_boxes: Union[torch.Tensor, List[torch.Tensor]] = None,
        boxes_for_pos_emb: Union[torch.Tensor, List[torch.Tensor]] = None,
        vt_boxes_for_pos_emb: Union[torch.Tensor, List[torch.Tensor]] = None,
        add_pos_embed: bool = True,
    ) -> torch.Tensor:
        """Extract object prompts from auxiliary features and optionally fuse tower features.

        Args:
            multi_level_features: Auxiliary feature maps in ``[N, C, H, W]`` format.
            boxes: Auxiliary RoIs in ``(x1, y1, x2, y2)`` format.
            vt_multi_level_features: Optional main vision-tower feature maps.
            vt_boxes: RoIs corresponding to main vision-tower feature maps.
            boxes_for_pos_emb: Auxiliary RoIs retained for positional encoding.
            vt_boxes_for_pos_emb: Vision-tower RoIs retained for positional encoding.
            add_pos_embed: Whether to add positional embeddings for this call.
        Returns:
            Object features with shape ``[1, num_rois, channels]``.
        """
        if boxes_for_pos_emb is None:
            boxes_for_pos_emb = copy.deepcopy(boxes)
        if vt_boxes_for_pos_emb is None:
            vt_boxes_for_pos_emb = copy.deepcopy(vt_boxes)

        if self.use_vt_object_feature_only:
            out_box_feat = self.extract_vt_object_feature(vt_multi_level_features, vt_boxes)

            if self.add_pos_embedding and add_pos_embed:
                pos_boxes = vt_boxes_for_pos_emb[0]  # (N, 4)
                pos_boxes = pos_boxes.to(out_box_feat.dtype)
                vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                original_img_width = vt_max_width / self.vision_tower_spatial_scale
                original_img_height = vt_max_height / self.vision_tower_spatial_scale
                pos_boxes[:, [0, 2]] = pos_boxes[:, [0, 2]] / original_img_width
                pos_boxes[:, [1, 3]] = pos_boxes[:, [1, 3]] / original_img_height
                # convert from xyxy to cx, cy, w, h
                pos_boxes[:, 2] = pos_boxes[:, 2] - pos_boxes[:, 0]
                pos_boxes[:, 3] = pos_boxes[:, 3] - pos_boxes[:, 1]
                pos_boxes[:, 0] = pos_boxes[:, 0] + pos_boxes[:, 2] / 2
                pos_boxes[:, 1] = pos_boxes[:, 1] + pos_boxes[:, 3] / 2
                pos_embed = gen_sineembed_for_position(pos_boxes.unsqueeze(0), self.pos_embedding_dim // 4)
                out_box_feat = out_box_feat + pos_embed
            return out_box_feat
        
        boxes[0] = boxes[0].float()
        
        if self.use_simpleFPN_for_vt_aux:
            multi_level_features = self.simple_fpn_aux(multi_level_features)

            roi_features_per_level = []
            # Use the scale associated with each auxiliary FPN level.
            feature_strides = [ self.simple_fpn_stride_aux / self.simple_fpn_scale_factors_aux[level_idx] for level_idx in range(len(self.simple_fpn_scale_factors_aux))]
            
            for level_idx, level_feature in enumerate(multi_level_features):
                current_spatial_scale = 1.0 / feature_strides[level_idx]
                
                level_roi_feat = roi_align(
                    level_feature.float(),
                    boxes,
                    output_size=self.output_size,
                    spatial_scale=current_spatial_scale,
                    aligned=self.roi_algined
                )
                
                level_roi_feat = level_roi_feat.mean(dim=(2, 3))
                roi_features_per_level.append(level_roi_feat)
            
            out_box_feat = torch.cat(roi_features_per_level, dim=1).unsqueeze(0)
        else:
            # Without an FPN, resize all levels to a common grid before concatenation.
            concat_multi_level_feature = []
            max_height = max([feature.shape[2] for feature in multi_level_features])
            max_width = max([feature.shape[3] for feature in multi_level_features])
            
            for level, feature in enumerate(multi_level_features):
                if feature.shape[-2] != max_height or feature.shape[-1] != max_width:
                    concat_multi_level_feature.append(
                        F.interpolate(
                            feature.float(),
                            size=(max_height, max_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                else:
                    concat_multi_level_feature.append(feature.float())
            concat_multi_level_feature = torch.cat(concat_multi_level_feature, dim=1)

            out_box_feat = roi_align(
                concat_multi_level_feature,
                boxes,
                output_size=self.output_size,
                spatial_scale=self.spatial_scale,
                aligned=self.roi_algined
            )
            
            # Pool spatial RoI features and restore the expected batch dimension.
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )

        if self.apply_object_layer_norm:
            out_box_feat = self.aux_object_norm.float()(out_box_feat)
        
        if self.use_vision_tower_object_feature:
            out_box_vt_feat = self.extract_vt_object_feature(vt_multi_level_features, vt_boxes)
            if self.apply_object_layer_norm:
                out_box_vt_feat = self.vt_object_norm.float()(out_box_vt_feat)
            if self.object_feature_combination in ['mean', 'mean_aux_pos']:
                # Mean fusion requires both branches to share the same embedding size.
                out_box_feat = (out_box_feat + out_box_vt_feat) / 2
            elif self.object_feature_combination in ['concat', 'concat_aux_pos']:
                if self.use_separate_mlp_for_object:
                    original_vt_dtype = out_box_vt_feat.dtype
                    original_aux_dtype = out_box_feat.dtype
                    out_box_vt_feat = self.object_mlp(out_box_vt_feat.to(self.object_mlp[0].weight.dtype)).to(original_vt_dtype)
                    out_box_feat = self.aux_object_mlp(out_box_feat.to(self.aux_object_mlp[0].weight.dtype)).to(original_aux_dtype)
                out_box_feat = torch.cat([out_box_feat, out_box_vt_feat], dim=-1)
        if self.add_pos_embedding and add_pos_embed:
            if self.add_pos_embedding:
                # Prefer vision-tower boxes when their features define the object prompt.
                if self.use_vision_tower_object_feature and vt_boxes is not None and self.object_feature_combination not in ['concat_aux_pos', 'mean_aux_pos']:
                    pos_boxes = vt_boxes_for_pos_emb[0]  # (N, 4)
                    # Infer input-image dimensions from the tower feature-map resolution.
                    vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                    vt_spatial_scale = self.vision_tower_spatial_scale
                    
                    original_img_width = vt_max_width / vt_spatial_scale
                    original_img_height = vt_max_height / vt_spatial_scale
                else:
                    # Otherwise, derive positions from auxiliary-feature RoIs.
                    max_width = max([feature.shape[3] for feature in multi_level_features])
                    max_height = max([feature.shape[2] for feature in multi_level_features])
                    pos_boxes = boxes_for_pos_emb[0]  # (N, 4)
                    original_img_width = max_width / self.spatial_scale
                    original_img_height = max_height / self.spatial_scale
                
                pos_boxes = pos_boxes.to(out_box_feat.dtype)
                pos_boxes[:, [0, 2]] = pos_boxes[:, [0, 2]] / original_img_width
                pos_boxes[:, [1, 3]] = pos_boxes[:, [1, 3]] / original_img_height
                # Convert normalized xyxy boxes to center-x, center-y, width, height.
                pos_boxes[:, 2] = pos_boxes[:, 2] - pos_boxes[:, 0]
                pos_boxes[:, 3] = pos_boxes[:, 3] - pos_boxes[:, 1]
                pos_boxes[:, 0] = pos_boxes[:, 0] + pos_boxes[:, 2] / 2
                pos_boxes[:, 1] = pos_boxes[:, 1] + pos_boxes[:, 3] / 2
                pos_embed = gen_sineembed_for_position(
                    pos_boxes.unsqueeze(0), self.pos_embedding_dim // 4
                )
                out_box_feat = out_box_feat + pos_embed

        return out_box_feat
