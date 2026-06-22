#!/usr/bin/env bash
# Compile best.onnx -> best.hef for the Raspberry Pi AI HAT (Hailo-8).
#
# RUN THIS ON AN x86_64 LINUX MACHINE (or Colab) that has the Hailo
# Dataflow Compiler (DFC) installed. It will NOT run on the Pi (ARM).
#
# Prereqs on the x86 box:
#   - Python venv with the Hailo DFC wheel installed (hailo_sdk_client importable).
#     Get the wheel from https://hailo.ai/developer-zone/  (free account).
#     Match the DFC version to the Pi's HailoRT runtime: 4.23.x.
#   - best.onnx and the IMG_*.JPG files from this repo present.
#   - python3 -m pip install opencv-python-headless numpy
#
# Usage:
#   python3 hailo/prepare_calib.py        # makes hailo/calib.npy
#   bash   hailo/compile_stopsign.sh
set -euo pipefail

HW_ARCH="hailo8"                  # Pi AI HAT (full Hailo-8). Use hailo8l for the 8L HAT.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
ONNX="$REPO/best.onnx"
CALIB="$HERE/calib.npy"
OUTDIR="$HERE/out"
NAME="stop_sign"

mkdir -p "$OUTDIR"

[ -f "$ONNX" ]  || { echo "Missing $ONNX (export it on the Pi: yolo export ... format=onnx opset=11)"; exit 1; }
[ -f "$CALIB" ] || { echo "Missing $CALIB — run: python3 hailo/prepare_calib.py"; exit 1; }
python3 -c "import hailo_sdk_client" 2>/dev/null || {
  echo "hailo_sdk_client not importable — install the Hailo Dataflow Compiler wheel first."; exit 1; }

echo "==> 1/3 Parse ONNX -> HAR"
# YOLOv8's DFL box-decode head (Sub/Add/Transpose on the detect output) is not
# Hailo-supported, so cut the graph before it. The remaining decode (anchor +
# DFL -> xywh) is done on the host. -y auto-accepts the parser's end-node split.
hailo parser onnx "$ONNX" \
  --hw-arch "$HW_ARCH" \
  --end-node-names /model.22/Sigmoid /model.22/dfl/Reshape \
  -y \
  --har-path "$OUTDIR/${NAME}.har"

echo "==> 2/3 Optimize/quantize (calibrated on repo stop-sign images)"
hailo optimize "$OUTDIR/${NAME}.har" \
  --hw-arch "$HW_ARCH" \
  --calib-set-path "$CALIB" \
  --output-har-path "$OUTDIR/${NAME}_optimized.har"

echo "==> 3/3 Compile -> HEF"
hailo compiler "$OUTDIR/${NAME}_optimized.har" \
  --hw-arch "$HW_ARCH" \
  --output-dir "$OUTDIR"

echo
echo "Done. Copy the .hef back to the Pi:"
ls -l "$OUTDIR"/*.hef
echo "Then on the Pi:  hailortcli run <that>.hef    (or wire it into HailoRT in Python)"
