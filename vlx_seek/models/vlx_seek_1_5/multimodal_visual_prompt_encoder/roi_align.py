"""RoI Align with native Ascend support and a CPU fallback for MPS."""

import torch
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align as torchvision_roi_align


class _NPURoIAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, output_size, spatial_scale, sampling_ratio, aligned):
        import torch_npu

        ctx.save_for_backward(rois)
        ctx.input_shape = input.shape
        ctx.output_size = output_size
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.roi_end_mode = 3 if aligned else 0
        return torch_npu.npu_roi_align(
            input, rois, spatial_scale, *output_size, sampling_ratio, ctx.roi_end_mode
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        import torch_npu

        (rois,) = ctx.saved_tensors
        height, width = ctx.output_size
        # The backward API takes width before height, unlike the forward API.
        grad_input = torch_npu.npu_roi_alignbk(
            grad_output.contiguous(), rois, ctx.input_shape, width, height,
            ctx.spatial_scale, ctx.sampling_ratio, ctx.roi_end_mode
        )
        return grad_input, None, None, None, None, None


def roi_align(
    input,
    boxes,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=-1,
    aligned=False,
):
    """Keep torchvision semantics on backends without native torchvision ops."""
    if input.device.type == "mps":
        if isinstance(boxes, (list, tuple)) and len(boxes) == 0:
            return input.reshape(-1)[:0].reshape(0, input.shape[1], *_pair(output_size))
        # torchvision has no native MPS ROIAlign kernel. FP32 also supports
        # half-precision feature maps on the CPU reference implementation.
        if isinstance(boxes, (list, tuple)):
            boxes = [box.to(device="cpu", dtype=torch.float32) for box in boxes]
        else:
            boxes = boxes.to(device="cpu", dtype=torch.float32)
        result = torchvision_roi_align(
            input.to(device="cpu", dtype=torch.float32), boxes, output_size,
            spatial_scale, sampling_ratio, aligned
        )
        return result.to(device=input.device, dtype=input.dtype)
    if input.device.type != "npu":
        return torchvision_roi_align(
            input, boxes, output_size, spatial_scale, sampling_ratio, aligned
        )

    output_size = _pair(output_size)
    if isinstance(boxes, (list, tuple)):
        rois = []
        for batch_index, box in enumerate(boxes):
            if box.ndim != 2 or box.shape[1] != 4:
                raise ValueError("Each boxes tensor must have shape [K, 4].")
            batch_ids = box.new_full((box.shape[0], 1), batch_index)
            rois.append(torch.cat((batch_ids, box), dim=1))
        rois = torch.cat(rois, dim=0) if rois else input.new_empty((0, 5))
    else:
        rois = boxes
    if rois.ndim != 2 or rois.shape[1] != 5:
        raise ValueError("RoIs must have shape [K, 5], including batch indices.")
    rois = rois.to(device=input.device, dtype=torch.float32).contiguous()

    if rois.shape[0] == 0:
        # Preserve the gradient connection without invoking an empty NPU kernel.
        return input.reshape(-1)[:0].reshape(0, input.shape[1], *output_size)

    # Ascend uses 0 for adaptive sampling; torchvision accepts any value <= 0.
    result = _NPURoIAlign.apply(
        input.float().contiguous(), rois, output_size,
        spatial_scale, max(sampling_ratio, 0), aligned
    )
    return result.to(dtype=input.dtype)
