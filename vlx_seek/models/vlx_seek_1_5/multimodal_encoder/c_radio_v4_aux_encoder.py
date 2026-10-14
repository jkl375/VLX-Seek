import torch
from .base_encoder import AbsVisionTower
from .c_radio_v4.hf_model import RADIOConfig, RADIOModel
from transformers import CLIPImageProcessor


class CRadioV4AuxEncoder(AbsVisionTower):
    """Aux vision tower that wraps NVIDIA's C-RADIOv4 model.

    Mimics the interface of other AuxEncoder so that it can be plugged
    into ``build_vision_tower_aux`` and consumed by the multi-level ROI visual
    prompt module (which expects ``forward(images) -> List[{"image_features": [...]}]``
    where each feature map has shape ``(B, C, H, W)``).
    """

    def __init__(self, vision_tower_name, args, delay_load=False, image_size=1024, aspect_ratio=None):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.image_size = image_size
        self.aspect_ratio = aspect_ratio
        self.out_indices = None
        self.args = args
        self.load_model()

    def load_model(self, is_train=False, image_size=None, aspect_ratio=None):
        aux_config = getattr(self.args, "vision_aux_config", None)
        if aux_config is None:
            raise ValueError(
                "vision_aux_config is required to initialize the C-RADIOv4-SO400M "
                "auxiliary vision tower."
            )
        if getattr(self.args, "mm_vitdet_window_size", None) is not None:
            raise ValueError("VitDet is not supported by the minimal C-RADIOv4 runtime.")

        if hasattr(aux_config, "to_dict"):
            aux_config = aux_config.to_dict()
        self.image_tower = RADIOModel(RADIOConfig(**aux_config))

        self.image_processor = CLIPImageProcessor(
            do_resize=False,
            do_center_crop=False,
            do_rescale=True,
            do_normalize=False,
            do_convert_rgb=True,
            resample=3,
        )

        is_meta_initialized = any(
            parameter.is_meta for parameter in self.image_tower.parameters()
        ) or any(buffer.is_meta for buffer in self.image_tower.buffers())
        if not is_meta_initialized:
            self.image_tower.to(dtype=torch.bfloat16, device=self.device)
        self.is_loaded = True

        if not is_meta_initialized:
            backbone_dtype = next(self.image_tower.radio_model.model.parameters()).dtype
            conditioner = self.image_tower.radio_model.input_conditioner
            if conditioner is not None and not isinstance(conditioner, torch.nn.Identity):
                conditioner.to(dtype=torch.float32)   # keep norm_mean / norm_std in fp32
                conditioner.dtype = backbone_dtype    # cast output to backbone dtype

        if not hasattr(self.image_tower.config, "hidden_size"):
            self.image_tower.config.hidden_size = self.image_tower.radio_model.model.embed_dim

        num_layers = len(self.image_tower.radio_model.model.blocks)
        self.out_indices = [
            round((i + 1) * num_layers / 4) - 1 for i in range(4)
        ]
        self.out_indices[-1] = num_layers - 1

    def set_grad_checkpointing(self, enable=True):
        backbone = self.image_tower.radio_model.model
        if hasattr(backbone, 'set_grad_checkpointing'):
            backbone.set_grad_checkpointing(enable)
        backbone.grad_checkpointing = enable
    
    def _extract_feature_map(self, intermediates):
        """``intermediates`` is a list of tensors with shape ``(B, C, H, W)``.

        Returns the structure expected by the downstream visual-prompt module.
        """
        if not isinstance(intermediates, (list, tuple)):
            intermediates = [intermediates]
        feature_maps = [feat for feat in intermediates]
        return {"image_features": feature_maps}

    def _resize_to_supported(self, image):
        """Resize ``image`` to the nearest H/W that are multiples of
        ``min_resolution_step`` (patch_size, or patch_size * window_size when
        vitdet is enabled). Uses RADIO's own rounding helper so we stay
        consistent with the model's assumptions.
        """
        radio = self.image_tower.radio_model
        h, w = image.shape[-2], image.shape[-1]
        target = radio.get_nearest_supported_resolution(h, w)
        if target.height == h and target.width == w:
            return image
        return torch.nn.functional.interpolate(
            image,
            size=(target.height, target.width),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

    def _forward_single(self, image):
        image = image.to(device=self.device, dtype=self.dtype)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        conditioner = self.image_tower.radio_model.input_conditioner
        if conditioner is not None and not isinstance(conditioner, torch.nn.Identity):
            conditioner.dtype = self.dtype

        image = self._resize_to_supported(image)

        intermediates = self.image_tower.radio_model.forward_intermediates(
            image,
            indices=self.out_indices,
            return_prefix_tokens=False,
            norm=False,
            output_fmt='NCHW',
            intermediates_only=True,
        )
        return self._extract_feature_map(intermediates)

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_features.append(self._forward_single(image))
            return image_features
        else:
            return self._forward_single(images)

    @property
    def dtype(self):
        for p in self.image_tower.radio_model.model.parameters():
            return p.dtype
        return torch.float32

    @property
    def device(self):
        return next(self.image_tower.radio_model.model.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.image_tower.radio_model.model.embed_dim
