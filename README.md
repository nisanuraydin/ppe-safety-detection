# PPE Safety Detection

This is an AI-based PPE detection project developed with Streamlit and YOLO.
It detects personal protective equipment in workplace images and videos,
including people, helmets, safety vests, and unprotected heads.

## Features

- Image and video analysis
- Webcam and online-stream support
- Helmet and safety-vest checks
- Custom danger-zone selection
- Violation warnings and snapshot saving
- A simple dashboard with Excel report export
- GPU support when CUDA is available

## Models

The application uses two trained models:

- `runs/detect/train/weights/best.pt`
  - Classes: `head`, `helmet`, `person`
- `runs/detect/train-2/weights/best.pt`
  - Classes: `boots`, `gloves`, `helmet`, `human`, `vest`

The detections from both models are combined to check whether a person is
wearing a helmet and safety vest.

## Datasets

The helmet model was trained with the Hard Hat dataset, and the PPE model was
trained with the Construction PPE Detection dataset.

More information about the datasets can be found in
[README.dataset.txt](README.dataset.txt) and
[README.roboflow.txt](README.roboflow.txt).

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/nisanuraydin/ppe-safety-detection.git
cd ppe-safety-detection
```

Create a virtual environment and install the dependencies.

### Windows

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

For NVIDIA GPU support:

```bash
pip install -r requirements-gpu.txt
```

The GPU requirements currently use the CUDA 12.1 version of PyTorch. If CUDA is
not available, the application runs on the CPU.

## Running the application

```bash
python -m streamlit run app.py
```

The application includes separate tabs for:

- Photo analysis
- Video analysis
- Camera and live-stream analysis
- Danger-zone selection
- Dashboard
- Settings

## Live sources

The camera tab supports a local webcam, uploaded videos, direct RTSP/HLS/MP4
URLs, and public YouTube streams.

Only use cameras and streams that you have permission to access. Some online
streams may stop working when their source URL changes.

## Project structure

```text
ppe-safety-detection/
├── app.py
├── app_helpers.py
├── data.yaml
├── dataset_vest/
├── runs/detect/
├── tests/
├── violations/
└── requirements.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Notes

- Detection results can change depending on lighting, camera angle, distance,
  and video quality.
- The danger zone is rectangular and is saved only for the current session.
- Dashboard and violation data are also kept only during the active Streamlit
  session.
- This is a portfolio project and should not be used as the only workplace
  safety control.

## License

This project is available under the [MIT License](LICENSE).
