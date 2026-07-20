import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

DEFAULT_MODEL_PATH = "runs/detect/train/weights/best.pt"
FALLBACK_MODELS = ["yolov8n.pt", "yolo26n.pt"]

st.set_page_config(page_title="İş Güvenliği Görüntü Analizi", page_icon="🦺", layout="wide")
st.title("🦺 İş Güvenliği Görüntü Analizi")
st.markdown(
    "Bu uygulama, iş güvenliği görsellerini ve videolarını analiz ederek koruyucu ekipman kullanımını tespit etmeye yardımcı olur."
)


def select_default_model():
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    for fallback in FALLBACK_MODELS:
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError(
        "Varsayılan model bulunamadı. Lütfen proje dizininde 'runs/detect/train/weights/best.pt' veya 'yolov8n.pt' ya da 'yolo26n.pt' dosyalarından birini bulundurun."
    )


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def save_uploaded_file(uploaded_file, suffix):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        return temp_file.name
    except Exception as exc:
        raise RuntimeError("Yüklenen dosya geçici olarak kaydedilemedi.") from exc


def annotate_video(input_path, output_path, conf):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı. Lütfen desteklenen bir video dosyası yükleyin.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video çözünürlüğü okunamadı.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Video kaydedici başlatılamadı.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    progress_bar = st.progress(0)
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
            writer.write(annotated)

            frame_index += 1
            if frame_count:
                progress_bar.progress(min(frame_index / frame_count, 1.0))

        return output_path
    finally:
        cap.release()
        writer.release()
        progress_bar.empty()


def summarize_detections(result, model_names):
    counts = {}
    for box in result.boxes:
        cls_name = model_names[int(box.cls)]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts
st.sidebar.header("Ayarlar")
st.sidebar.write(
    "Bu uygulama, iş güvenliği amaçlı görüntü ve video verisini analiz eder. "
    "Varsayılan model otomatik olarak yüklenir."
)

try:
    selected_model_path = select_default_model()
    model = load_model(selected_model_path)
    st.sidebar.write(f"Yüklenecek model: `{selected_model_path}`")
    try:
        class_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        st.sidebar.write("**Model sınıfları:**")
        st.sidebar.write(", ".join(class_names))
    except Exception:
        pass
except Exception as exc:
    st.error(f"Model yüklenirken hata oluştu: {exc}")
    st.stop()

st.caption("💡 Bu model için istatistiksel olarak en dengeli eşik: **0.34** (F1 skoruna göre hesaplandı)")
conf = st.slider("Güven eşiği", 0.1, 0.9, 0.34, 0.05)

st.markdown("---")
image_tab, video_tab, camera_tab = st.tabs(["Fotoğraf", "Video", "Kamera"])

with image_tab:
    st.subheader("Fotoğraf Analizi")
    uploaded_file = st.file_uploader("Fotoğraf seç", type=["jpg", "jpeg", "png"], key="image_input")
    if uploaded_file is not None:
        if st.button("Analiz Et", key="analyze_image"):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                with st.spinner("Görsel analiz ediliyor..."):
                    results = model.predict(image, conf=conf)
                    result = results[0]
                    annotated = result.plot()[:, :, ::-1]
                    st.image(annotated, caption="Tespit sonucu", use_container_width=True)

                    if len(result.boxes) == 0:
                        st.info("Bu görüntüde tespit edilen herhangi bir nesne yok.")
                    else:
                        st.write("**Tespit edilenler:**")
                        for box in result.boxes:
                            cls_name = model.names[int(box.cls)]
                            st.write(f"- **{cls_name}** — güven: {float(box.conf):.0%}")
                        summary = summarize_detections(result, model.names)
                        st.write("**Tespit özeti:**")
                        for name, count in summary.items():
                            st.write(f"- {name}: {count}")
            except Exception as exc:
                st.error(f"Görsel analizinde hata oluştu: {exc}")

with video_tab:
    st.subheader("Video Analizi")
    uploaded_video = st.file_uploader("Video seç", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_input")
    if uploaded_video is not None:
        if st.button("Videoyu Analiz Et", key="analyze_video"):
            with st.spinner("Video analiz ediliyor. Bu işlem uzun sürebilir..."):
                try:
                    video_suffix = f".{uploaded_video.name.split('.')[-1]}"
                    video_path = save_uploaded_file(uploaded_video, suffix=video_suffix)
                    annotated_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    annotated_video_path = annotate_video(video_path, annotated_video_path, conf=conf)

                    st.success("Video analizi tamamlandı.")
                    st.video(annotated_video_path)
                    mime_type = "video/mp4" if annotated_video_path.endswith(".mp4") else "video/x-msvideo"
                    with open(annotated_video_path, "rb") as out_file:
                        st.download_button(
                            "Analiz edilmiş videoyu indir",
                            out_file.read(),
                            file_name=f"annotated_video{'.mp4' if annotated_video_path.endswith('.mp4') else '.avi'}",
                            mime=mime_type,
                        )
                except Exception as exc:
                    st.error(f"Video analizinde hata oluştu: {exc}")

with camera_tab:
    st.subheader("Kamera Analizi")
    if st.button("Kameradan bir kare al", key="capture_webcam"):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Kameraya erişilemedi.")

            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Kamera görüntüsü alınamadı.")

            with st.spinner("Kamera görüntüsü işleniyor..."):
                results = model.predict(frame, conf=conf, verbose=False)
                result = results[0]
                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Kamera görüntüsü", use_container_width=True)
                if len(result.boxes) == 0:
                    st.info("Bu karede herhangi bir tespit yok.")
                else:
                    summary = summarize_detections(result, model.names)
                    st.write("**Tespit özeti:**")
                    for name, count in summary.items():
                        st.write(f"- {name}: {count}")
        except Exception as exc:
            st.error(f"Kamera analizinde hata oluştu: {exc}")