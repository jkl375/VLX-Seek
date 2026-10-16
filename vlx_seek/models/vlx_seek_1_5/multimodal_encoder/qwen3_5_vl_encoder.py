import os
from copy import deepcopy
import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from torchvision.transforms import ToPILImage
from typing import Optional, List


class Qwen3_5_VlVisionTower(nn.Module):
    """Vision-tower wrapper for extracting Qwen3.5-VL image embeddings."""

    def __init__(self, image_tower, args, delay_load=False, min_pixels=56*56, max_pixels=2048*2048):
        super().__init__()

        self.is_loaded = False
        self.args = args
        model_name = getattr(args, 'name_or_path', '').split('/')[-1].lower()
        # VLX-Seek checkpoints provide a vision config, while standalone
        # checkpoints are loaded directly from the configured vision tower.
        self.name_or_path = args.name_or_path if 'vlx-seek' in model_name or 'vlx_seek' in model_name else None
        self.image_tower_name = image_tower
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.use_vision_tower_object_feature = getattr(args, 'mm_use_vision_tower_object_feature', False)
        self.use_unmerged_features = False
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.delay_load = delay_load
        
        self.load_model()

    def load_model(self, image_size=336, is_train=False):
        attn_implementation = getattr(self.args, "_attn_implementation", None) or "flash_attention_2"
        if self.name_or_path is not None:
            # Build from the checkpoint configuration; weights are loaded by
            # the parent VLX-Seek model.
            self.visual = Qwen3_5VisionModel._from_config(self.args.vision_config, attn_implementation=attn_implementation, dtype=torch.bfloat16)
            self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.name_or_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        else:
            # Load an independent Hugging Face vision checkpoint with weights.
            self.visual, loading_info = Qwen3_5VisionModel.from_pretrained(self.image_tower_name, attn_implementation=attn_implementation, dtype=torch.bfloat16, output_loading_info=True)
            self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.image_tower_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        self.is_loaded = True

    def convert_image_format(self, image):
        """Convert a tensor image into processor-ready vision inputs."""
        pil_image = ToPILImage()(image)
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        return inputs['pixel_values'], inputs['image_grid_thw']

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """Run the vision model and split embeddings back into individual images."""
        pixel_values = pixel_values.type(self.dtype).to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)
        vision_output = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True
        )
        image_embeds = vision_output.pooler_output
        # The pooled sequence contains all images concatenated along tokens.
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds

        return image_embeds, vision_output
    
    def get_multi_level_features(self, image_embeds: torch.Tensor, image_grid_thw: torch.LongTensor):
        """Reshape the final token sequence into a spatial feature map."""
        image_grid_thw = image_grid_thw.cpu().tolist()
        if len(image_grid_thw) == 1 and isinstance(image_grid_thw[0], list):
            image_grid_thw = image_grid_thw[0]
        T, H, W = image_grid_thw
        if not self.use_unmerged_features:
            # Pooler output has already applied spatial token merging.
            H = H // self.visual.spatial_merge_size
            W = W // self.visual.spatial_merge_size
        multi_level_features = []
        last_feature = image_embeds.view(T, H, W, -1).permute(0, 3, 1, 2).contiguous()
        multi_level_features.append(last_feature)
        return multi_level_features

    def forward(self, images, image_grid_thws=[]):
        """Extract per-image sequence features and spatial feature maps."""
        if type(images) is list:
            image_features = []
            multi_level_features_list = []
            output_image_grid_thws = []

            for i, image in enumerate(images):
                if image_grid_thws is None or len(image_grid_thws) == 0:
                    image, image_grid_thw = self.convert_image_format(image=image)
                else:
                    image_grid_thw = image_grid_thws[i]
                image_embeds, vision_output = self.get_image_features(image, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)
                
                # Preserve a batch dimension for the downstream projector.
                image_feature = image_embeds.unsqueeze(0).to(self.dtype)

                if self.use_unmerged_features:
                    image_embeds = vision_output.last_hidden_state
                multi_level_features = self.get_multi_level_features(image_embeds, image_grid_thw)

                image_features.append(image_feature)
                output_image_grid_thws.append(image_grid_thw)
                multi_level_features_list.append(multi_level_features)
        else:
            raise NotImplementedError

        return image_features, output_image_grid_thws, multi_level_features_list
    

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.visual.dtype

    @property
    def device(self):
        return self.visual.device

    @property
    def config(self):
        if self.is_loaded:
            return self.visual.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.out_hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

