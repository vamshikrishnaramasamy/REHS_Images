# Running the stop-sign model on the Raspberry Pi AI HAT (Hailo-8)

The Pi runs `best.pt` on CPU at **~6 FPS**. On the Hailo-8 AI HAT a yolov8-class
model runs at **~300+ FPS** (benchmarked: yolov8s = 310 FPS; `best.pt` is yolov8n,
so even faster). To get there, `best.pt` must be compiled to a `.hef`.

## Why not just do it on the Pi?

The Hailo **Dataflow Compiler (DFC)** that produces `.hef` files is **x86_64-only**.
It cannot run on the Pi (ARM). The Pi only has the *runtime* (HailoRT 4.23.0), which
runs an already-compiled `.hef`. So the compile step happens elsewhere, once.

## Pipeline

```
best.pt  --(on Pi, done)-->  best.onnx  --(x86 DFC)-->  best.hef  --(copy to Pi)-->  ~300 FPS
```

`best.onnx` is already exported and committed (opset 11, 640x640, NMS-free).

## Where to compile

| Option        | Notes |
|---------------|-------|
| **Colab**     | Free x86 Ubuntu, ~12 GB RAM — enough for yolov8n. Must upload the gated DFC wheel. Session resets, so one-shot. |
| **Local x86** | Any Intel/AMD box, **16 GB RAM recommended**, Ubuntu 20.04/22.04 (Docker image easiest). GPU optional. More reliable. |

Either way, get the **Dataflow Compiler wheel matching HailoRT 4.23.x** from
<https://hailo.ai/developer-zone/> (free account) and install it into a Python env
so `import hailo_sdk_client` works.

## Steps (on the x86 box / Colab)

```bash
# 0. clone this repo so best.onnx + IMG_*.JPG are present
# 1. install the Hailo DFC wheel (from Developer Zone) into your venv
pip install opencv-python-headless numpy

# 2. build calibration set from the repo's stop-sign photos
python3 hailo/prepare_calib.py        # -> hailo/calib.npy

# 3. parse -> optimize/quantize -> compile
bash hailo/compile_stopsign.sh        # -> hailo/out/stop_sign.hef
```

## Back on the Pi

```bash
# quick sanity / FPS check
hailortcli run hailo/out/stop_sign.hef

# benchmark throughput
hailortcli benchmark hailo/out/stop_sign.hef
```

For live detection wire the `.hef` into HailoRT's Python API (`hailo_platform`),
feeding frames from Picamera2 — same camera setup as `live_stop_sign_pi.py`,
but inference goes to the HAT instead of torch-on-CPU.

> Note: use `--hw-arch hailo8` (this HAT is a full Hailo-8). Use `hailo8l` only if
> you have the Hailo-8L HAT.
