# PPE Safety Detection

A Streamlit application that analyzes personal protective equipment (PPE) use in factory and worksite images. It uses two YOLO models to detect helmets, safety vests, people, and unprotected heads.

## Features

- Image and video analysis
- Local webcam, uploaded video, RTSP/HLS/MP4 URL, and public YouTube live-stream sources
- Per-person helmet and vest compliance checks
- Configurable danger zone, time-based critical-violation alerts, and snapshot capture
- Live violation log, dashboard, and Excel report export
- CUDA-accelerated inference when an NVIDIA GPU is available

## Models and datasets

- `runs/detect/train/weights/best.pt`: helmet model (`head`, `helmet`, `person`)
- `runs/detect/train-2/weights/best.pt`: PPE model (`boots`, `gloves`, `helmet`, `human`, `vest`)
- `data.yaml`: configuration for the helmet dataset
- `dataset_vest/data.yaml`: configuration for the PPE dataset

See [README.dataset.txt](README.dataset.txt) and [README.roboflow.txt](README.roboflow.txt) for dataset source information.

## Setup

Windows and Python 3.11 are recommended.

```powershell
cd C:\Users\casper\OneDrive\Desktop\ppe-detection
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA-enabled PyTorch packages after the base dependencies:

```powershell
pip install -r requirements-gpu.txt
```

## Run the app

In VS Code, select this interpreter:

```text
.venv311\Scripts\python.exe
```

Then start the app:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run app.py
```

## Live sources

The Camera page supports the following sources:

- Your computer's local webcam
- An uploaded video file
- Direct `rtsp://`, HLS (`.m3u8`), or MP4 URLs
- Public YouTube video and live-stream URLs

Only use sources that you have permission to access. Some providers require a direct stream URL instead of a webpage URL, and their stream availability may change over time.

## Tests

Run the UI-independent danger-zone tests with:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv311\Scripts\python.exe -m pytest -q
```

## Limitations

- This is an educational portfolio prototype and is not a replacement for human safety supervision.
- Detection quality depends on the training data, camera angle, lighting, and video quality.
- Online-stream latency and reliability depend on the stream provider.
- Dashboard data is kept only for the active Streamlit session.

## License

This project is distributed under the [MIT License](LICENSE).
