import cv2
from ultralytics import YOLO
from app import combined_predict
from app_helpers import resolve_project_path

video_path = resolve_project_path('video_test.mp4')
helmet_model = YOLO(str(resolve_project_path('runs','detect','train','weights','best.pt')))
vest_model = YOLO(str(resolve_project_path('runs','detect','train-2','weights','best.pt')))

cap = cv2.VideoCapture(str(video_path))
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 10 != 0:
        frame_idx += 1
        continue
    annotated, summary, zone_violation = combined_predict(frame, 0.34, 0.585, helmet_model, vest_model, (0, 0, frame.shape[1], frame.shape[0]))
    print('frame', frame_idx, 'summary', summary, 'zone_violation', zone_violation)
    frame_idx += 1
cap.release()
