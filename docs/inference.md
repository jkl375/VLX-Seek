# VLX-Seek 1.5 Inference Guide

English | [中文](inference_zh.md)

This guide covers command-line and Python inference for VLX-Seek 1.5-10B. For the model overview and installation instructions, see the [project README](../README.md).

## Prerequisites

The provided pipeline currently requires:

- inference dependencies, installed locally with `requirements.txt` or provided by the Docker image described below;
- the VLX-Seek 1.5-10B checkpoint from [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B);
- the region detector checkpoint from [WeDetect](https://huggingface.co/fushh7/WeDetect)

You can load the VLX-Seek main model directly from Hugging Face:

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

When `--bbox-list` is omitted, `wedetect_base_uni.pth` is automatically downloaded to `resources/wedetect_base_uni.pth` if needed.

Alternatively, place local copies in the default directory:

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` accepts either a Hugging Face model identifier or a local checkpoint directory. The model architecture is provided by the local `vlx_seek` package in this repository.

Alternatively, set paths with `--model-path` and `--detector-checkpoint`.

### Hardware Selection

Use `VLX_SEEK_BACKEND` to select hardware for both the CLI and `VLXSeekWorker`:

| Value | Device |
| --- | --- |
| `auto` (default) | Detect NVIDIA CUDA first, then Ascend NPU, otherwise CPU |
| `nvidia` | NVIDIA GPU with CUDA |
| `ascend` | Ascend NPU; currently tested on 910B |
| `cpu` | CPU with SDPA attention |
| `apple` | Reserved for Apple M-series/MPS; not yet supported |

```bash
VLX_SEEK_BACKEND=nvidia python inference.py --task detection --text "orange; apple"
VLX_SEEK_BACKEND=ascend python inference.py --task detection --text "orange; apple"
```

An explicit `--device cuda:1`, `--device npu:0`, or Python API `device=` overrides the environment variable. Selecting an unavailable backend raises an error instead of silently switching hardware. Auto detection reports an unsupported-backend error on an MPS-only machine until Apple support is implemented; use `cpu` to select the CPU path explicitly. NVIDIA/CPU selection does not require NPU dependencies or change CUDA APIs.

### Optional: Faster Inference

The model uses Linear Attention layers. Without the corresponding acceleration kernels, you will see a warning like the following and fall back to a slower PyTorch implementation:

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

On NVIDIA GPUs, installing [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) enables the fast path. For NPU acceleration, see the [Ascend 910B instructions](#docker-inference-on-ascend-910b-aarch64) below.

## Docker Inference on NVIDIA x86 (e.g. A100)

Use the provided Docker environment on a Linux x86_64 host with an NVIDIA GPU such as A100. The host needs an NVIDIA driver, Docker, and NVIDIA Container Toolkit configured for GPU containers. The image includes the inference dependencies, including `flash-linear-attention` and `causal-conv1d`.

Host NVIDIA driver requirements:

| x86 GPU type | Minimum Driver Version |
| --- | --- |
| Data center GPUs such as A100 | 535 or later |
| Consumer GPUs such as RTX 4090 | 580 or later |

### Prepare the Image

Clone the repository if needed, then run all host-side commands from its root directory:

```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek
```

Choose one of the following:

- Build with [Dockerfile.nvidia_x86](../docker/Dockerfile.nvidia_x86):

  ```bash
  docker build -f docker/Dockerfile.nvidia_x86 -t vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 .
  ```

- Pull our prebuilt image:

  ```bash
  docker pull registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64
  ```

### Start and Enter the Container

Run the following command from the repository root. It selects host GPU `0` through `--gpus device=0`; change the index to use another GPU. For a locally built image, replace `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64` with `vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64`.

```bash
docker run -d --name seek-dev \
  --gpus device=0 \
  --shm-size=40g \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=nvidia \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 \
  sleep infinity
docker exec -it seek-dev bash
```

The command mounts the repository at `/workspace` and sets it as the working directory, making your host-side source code, local weights, input images, and saved results available inside the container. The container stays running so you can enter it again with `docker exec -it seek-dev bash`.

### Run Inference Inside the Container

```bash
python inference.py \
  --model-path omlab/VLX-Seek-1.5-10B \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

The first run requires access to Hugging Face to download the main model and, when proposals are needed, `resources/wedetect_base_uni.pth`. To use a local main-model checkpoint, place it in the host repository's `resources/VLX-Seek-1.5-10B/` directory and use `--model-path resources/VLX-Seek-1.5-10B` instead. The Docker image does not bundle model weights.

Results are printed as JSON, and the example visualization is saved to `demo/demo_image_result.png` in the mounted host repository. You can run the other CLI examples and Python API below from `/workspace` in the same container.

## Docker Inference on T5000 and GB10 (aarch64)

Host platform requirements:

| Platform | Requirement |
| --- | --- |
| Jetson series (T5000) | JetPack 7.0 or later; JetPack 7.2 recommended |
| NVIDIA Spark GB10 | NVIDIA Driver Version 580 or later |

Use the prebuilt image matching your aarch64 platform:

| Platform | Image |
| --- | --- |
| T5000 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64` |
| GB10 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64` |

The host needs the NVIDIA driver, Docker, and NVIDIA Container Toolkit with the `nvidia` runtime configured. Clone the repository as shown above and run all host-side commands from its root directory.

### T5000 Docker Run

Run from the repository root to start and enter the container:

```bash
docker run -d --name seek-dev \
  --runtime=nvidia \
  --restart=always \
  --shm-size=40g \
  -p 1233:8000 \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=nvidia \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64 \
  sleep infinity
docker exec -it seek-dev bash
```

### GB10 Docker Run

Run from the repository root to start and enter the container:

```bash
docker run -d --name seek-dev \
  --runtime=nvidia \
  --restart=always \
  --shm-size=40g \
  -p 1233:8000 \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=nvidia \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64 \
  sleep infinity
docker exec -it seek-dev bash
```

`docker run` pulls the selected image if it is not available locally. Both commands mount the repository at `/workspace`, publish container port `8000` on host port `1233`, and use the container name `seek-dev`. Stop and remove an existing container with that name before switching platforms.

Run the inference command in the NVIDIA x86 section from `/workspace` inside the container. Model downloads, local weight paths, and output paths work the same way as described there.

## Docker Inference on Ascend 910B (aarch64)

Use the prebuilt image on a Linux aarch64 host with an Ascend 910B NPU. The host needs Docker and a working Ascend driver; verify the device with `npu-smi info`. The image includes CANN 9.0.1, PyTorch 2.9.0, `torch_npu` 2.9.0.post4, Transformers 5.13.0, Ascend Triton, and NPU FLA dependencies. This inference path has been tested on a 910B4-1.

### Prepare the Image and Container

Clone the repository as shown above, then run the following commands from its root:

```bash
docker pull registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-910b-aarch64
docker run -d --name seek-dev \
  --entrypoint sleep \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=10g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e VLX_SEEK_BACKEND=ascend \
  registry.linker.cc/vlx/vlx-seek:py3.11-openeuler24.03-910b-aarch64 \
  infinity
docker exec -it seek-dev bash
```

Replace `/dev/davinci0` with the host device selected using `npu-smi info`, for example `/dev/davinci2`. Adjust the driver mount paths if your host uses a different layout. All Docker examples use the container name `seek-dev`; use another name if it is already occupied.

### Run Inference Inside the Container

Run from `/workspace`:

```bash
VLX_SEEK_BACKEND=ascend VLX_SEEK_NPU_FLA=full python inference.py \
  --model-path omlab/VLX-Seek-1.5-10B \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

The Ascend backend imports `torch_npu` and places both the proposal detector and VLX-Seek on NPU explicitly. It does not globally replace CUDA APIs. The image does not include model weights. The first run downloads the main model and, if needed, the proposal detector. To use local weights, set `--model-path resources/VLX-Seek-1.5-10B`. Results and the visualization are saved in the mounted repository as described above.

NPU FLA is enabled by default when its dependencies are available:

| `VLX_SEEK_NPU_FLA` | Behavior |
| --- | --- |
| `full` (default) | FLA prefill and token decoding |
| `decode` | PyTorch prefill and FLA token decoding |
| `off` | Original PyTorch linear attention |

Look for `NPU FLA enabled on 24 linear-attention layers` to confirm activation. Transformers may print `The fast path is not available` before this message during model construction. The NPU path uses native ROIAlign and retains PyTorch causal convolution; the upstream CUDA `causal-conv1d` package is not required.

The first run may take several minutes to compile Triton kernels. Measure performance after warmup; numerical differences from the PyTorch implementation are possible. See the [NPU FLA implementation and measurements (Chinese)](npu_fla_zh.md) for details.

## Region Proposals

Most region-level tasks first require candidate bounding boxes. Due to company policy, we are unable to release the internally trained OPN referenced in our blog. The repository integrates the open-source, lightweight **WeDetect-Base-Uni** detector as an alternative proposal generator.

There are two ways to provide proposals:

1. Omit `--bbox-list`. The CLI loads WeDetect-Uni and generates 100 proposals.
2. Use any detector of your choice and pass its proposals through `--bbox-list`.

Custom boxes must use original-image pixel coordinates in `[x1, y1, x2, y2]` format:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple" \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

The model supports up to 100 object-region tokens, so keep the proposal list at or below 100 boxes.

## Command-Line Usage

### Open-Vocabulary Detection

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

Multiple detection categories are separated by semicolons. Add `--lang zh` when using the built-in Chinese task templates.

### Referring Expression Grounding

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "the person in a white shirt"
```

### Object Counting

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "orange"
```

### Reasoning Detection

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "the woman wearing red clothes"
```

### Region Captioning

Region captioning requires both proposals and one or more zero-based proposal indexes:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]'
```

The indexes refer to the order of the boxes supplied through `--bbox-list`.

### Region OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]'
```

### General VQA

VQA operates on the full image and does not load a proposal model:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "What fruits are in the image?"
```

## Supported Tasks

| Task | `--text` | Proposals | Additional requirement |
| --- | --- | --- | --- |
| `detection` | One or more `;`-separated categories/descriptions | Required | None |
| `grounding_single` | A referring expression | Required | None |
| `counting` | A target description | Required | None |
| `reasoning_detection` | A target description or category | Required | None |
| `brief_region_caption` | Generated from selected regions | Required | `--target-region-indexes` |
| `region_ocr` | Generated from selected regions | Required | `--target-region-indexes` |
| `vqa` | A free-form question | Not used | None |

For proposal-dependent tasks, omitting `--bbox-list` automatically invokes the proposal detector.

## Common Options

| Option | Default | Description |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | Local checkpoint directory or Hugging Face model identifier, e.g. `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | Input image |
| `--lang` | `en` | Built-in task-template language: `en` or `zh` |
| `--bbox-list` | None | JSON list of pixel-coordinate proposal boxes |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | WeDetect-Uni checkpoint used when proposals are omitted; auto downloaded if the default path is missing |
| `--target-region-indexes` | None | JSON list of selected proposal indexes for captioning or OCR |
| `--max-new-tokens` | `2048` | Maximum completion length |
| `--temperature` | `0.0` | Sampling temperature; values above zero enable sampling. Temperature `0.0` is recommended for all detection-related tasks |
| `--no-visualize` | Off | Disable result-box visualization |
| `--visualization-output` | `<image_stem>_result.png` | Custom visualization path |

Run `python inference.py --help` for the complete CLI reference.

## Output

The command prints a result dictionary as formatted JSON:

```json
{
  "answer": "<ground>orange</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "orange"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` is the raw model response.
- `result_bbox_list` resolves generated `<objN>` references to the input proposal coordinates.
- `prompt_tokens` and `completion_tokens` report token counts.
- `visualization_path` is included when selected boxes are visualized.

VQA, captioning, or OCR answers may be free-form text and can have an empty `result_bbox_list`.

## Python API

`VLXSeekWorker` keeps the model loaded across requests. Proposal generation is handled by the CLI, so Python API callers should prepare their own boxes for region-level tasks.

```python
from PIL import Image

from vlx_seek_worker import VLXSeekWorker

worker = VLXSeekWorker(
    model_path="omlab/VLX-Seek-1.5-10B",
)
image = Image.open("demo/demo_image.jpg").convert("RGB")
boxes = [
    [12, 20, 180, 210],
    [190, 25, 430, 250],
]

detection = worker.detect(
    image,
    boxes,
    ["orange", "apple"],
    lang="en",
)

vqa = worker.predict(
    image,
    "What fruits are in the image?",
)
```

Other task wrappers are `ground`, `count`, `reasoning_detect`, `describe_region`, and `read_region_text`. Boxes are clipped to the input image bounds and must retain positive area after clipping.
