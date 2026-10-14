# export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PWD}/.cache/triton}"
# export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# mkdir -p "$TRITON_CACHE_DIR"

python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
