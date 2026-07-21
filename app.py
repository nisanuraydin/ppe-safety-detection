import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

HELMET_MODEL_PATH = "runs/detect/train/weights/best.pt"
VEST_MODEL_PATH = "runs/detect/train-2/weights/best.pt"

st.set_page_config(page_title="İş Güvenliği Görüntü Analizi", page_icon="🦺", layout="wide")
st.title("🦺 İş Güvenliği Görüntü Analizi")
st.markdown(
    "Bu uygulama, iş güvenliği görsellerini ve videolarını analiz ederek koruyucu ekipman kullanımını tespit etmeye yardımcı olur."
)


@st.cache_resource
def load_models():
    helmet_model = YOLO(HELMET_MODEL_PATH)
    vest_model = YOLO(VEST_MODEL_PATH)
    return helmet_model, vest_model


def save_uploaded_file(uploaded_file, suffix):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        return temp_file.name
    except Exception as exc:
        raise RuntimeError("Yüklenen dosya geçici olarak kaydedilemedi.") from exc


def point_in_zone(cx, cy, zone):
    """Bir noktanın (cx, cy) dikdörtgen bölgenin (x1,y1,x2,y2) içinde olup olmadığını kontrol eder."""
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


def get_absolute_zone(relative_zone, width, height):
    """
    Oransal (0-1 arası) bölge koordinatlarını, verilen görüntü/kare boyutuna göre
    gerçek piksel koordinatlarına çevirir. Bu sayede bölge, referans fotoğraftan
    farklı çözünürlükteki video/kamera karelerinde de doğru yerde çizilir.
    """
    if relative_zone is None:
        return None
    rx1, ry1, rx2, ry2 = relative_zone
    return (
        int(rx1 * width),
        int(ry1 * height),
        int(rx2 * width),
        int(ry2 * height),
    )


def combined_predict(image, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone=None):
    """
    İki modeli aynı görüntüde, KENDİ optimal güven eşikleriyle çalıştırır:
    - vest_model: boots, gloves, helmet, human, vest tespit eder (temel görsel bundan gelir)
    - helmet_model: sadece 'head' (baretsiz kafa = ihlal) bilgisini ekler, kırmızı kutu ile
    - danger_zone verilmişse: 'human' kutularının merkezi bölge içindeyse turuncu kutu + uyarı ekler

    Böylece 'helmet' sınıfı iki modelde de olduğu için tekrar/çakışma olmaz;
    helmet_model'den yalnızca ihlal (head) bilgisi alınır.
    """
    helmet_result = helmet_model.predict(image, conf=helmet_conf, verbose=False)[0]
    vest_result = vest_model.predict(image, conf=vest_conf, verbose=False)[0]

    annotated = vest_result.plot()

    head_class_id = None
    for idx, name in helmet_model.names.items():
        if name == "head":
            head_class_id = idx
            break

    summary = {}

    if head_class_id is not None:
        for box in helmet_result.boxes:
            if int(box.cls) == head_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"No-Helmet {float(box.conf):.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                summary["head (baretsiz)"] = summary.get("head (baretsiz)", 0) + 1

    for box in vest_result.boxes:
        cls_name = vest_model.names[int(box.cls)]
        summary[cls_name] = summary.get(cls_name, 0) + 1

    zone_violation = False

    if danger_zone is not None:
        zx1, zy1, zx2, zy2 = danger_zone
        # Bölgeyi görsel olarak çiz (sarı çerçeve)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
        cv2.putText(annotated, "DANGER ZONE", (zx1, max(zy1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        human_class_id = None
        for idx, name in vest_model.names.items():
            if name == "human":
                human_class_id = idx
                break

        if human_class_id is not None:
            for box in vest_result.boxes:
                if int(box.cls) == human_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if point_in_zone(cx, cy, danger_zone):
                        zone_violation = True
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 3)
                        cv2.putText(annotated, "DANGER ZONE VIOLATION", (x1, max(y1 - 8, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                        summary["danger_zone_violation"] = summary.get("danger_zone_violation", 0) + 1

    return annotated, summary, zone_violation


def annotate_video(input_path, output_path, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone_relative=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı. Lütfen desteklenen bir video dosyası yükleyin.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video çözünürlüğü okunamadı.")

    # Oransal bölgeyi, bu videonun gerçek çözünürlüğüne göre piksel koordinatına çevir
    danger_zone = get_absolute_zone(danger_zone_relative, width, height)

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

    # Bölge ihlali olan zaman aralıklarını tutmak için
    zone_intervals = []
    currently_in_violation = False
    violation_start_time = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated, _, zone_violation = combined_predict(
                frame, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
            )
            writer.write(annotated)

            current_time = frame_index / fps

            if danger_zone is not None:
                if zone_violation and not currently_in_violation:
                    # İhlal yeni başladı
                    currently_in_violation = True
                    violation_start_time = current_time
                elif not zone_violation and currently_in_violation:
                    # İhlal az önce bitti
                    currently_in_violation = False
                    zone_intervals.append((violation_start_time, current_time))

            frame_index += 1
            if frame_count:
                progress_bar.progress(min(frame_index / frame_count, 1.0))

        # Video ihlal ile bitiyorsa son aralığı da kapat
        if currently_in_violation:
            zone_intervals.append((violation_start_time, frame_index / fps))

        return output_path, zone_intervals
    finally:
        cap.release()
        writer.release()
        progress_bar.empty()


# --- Modelleri yükle ---
try:
    helmet_model, vest_model = load_models()
    st.sidebar.header("Ayarlar")
    st.sidebar.write("**Baret modeli sınıfları:**")
    st.sidebar.write(", ".join(helmet_model.names.values()))
    st.sidebar.write("**Yelek modeli sınıfları:**")
    st.sidebar.write(", ".join(vest_model.names.values()))
except Exception as exc:
    st.error(f"Model(ler) yüklenirken hata oluştu: {exc}")
    st.stop()

st.subheader("Güven Eşikleri")
st.caption("Her model kendi F1-Confidence eğrisine göre farklı bir optimal eşiğe sahip, bu yüzden ayrı ayarlanıyor.")

col1, col2 = st.columns(2)
with col1:
    st.caption("💡 Baret modeli için optimal: **0.34**")
    helmet_conf = st.slider("Baret modeli güven eşiği", 0.1, 0.9, 0.34, 0.01, key="helmet_conf")
with col2:
    st.caption("💡 Yelek modeli için optimal: **0.585**")
    vest_conf = st.slider("Yelek modeli güven eşiği", 0.1, 0.9, 0.585, 0.01, key="vest_conf")

st.markdown("---")
image_tab, video_tab, camera_tab, zone_tab = st.tabs(["Fotoğraf", "Video", "Kamera", "Tehlikeli Bölge"])

with image_tab:
    st.subheader("Fotoğraf Analizi")
    uploaded_file = st.file_uploader("Fotoğraf seç", type=["jpg", "jpeg", "png"], key="image_input")
    if uploaded_file is not None:
        if st.button("Analiz Et", key="analyze_image"):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                with st.spinner("Görsel analiz ediliyor..."):
                    relative_zone = st.session_state.get("danger_zone_relative")
                    danger_zone = get_absolute_zone(relative_zone, image.width, image.height)
                    annotated_bgr, summary, zone_violation = combined_predict(
                        image, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
                    )
                    annotated = annotated_bgr[:, :, ::-1]
                    st.image(annotated, caption="Tespit sonucu", use_container_width=True)

                    if zone_violation:
                        st.error("⚠️ Tehlikeli bölgede kişi tespit edildi!")

                    if len(summary) == 0:
                        st.info("Bu görüntüde tespit edilen herhangi bir nesne yok.")
                    else:
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
                    danger_zone_relative = st.session_state.get("danger_zone_relative")
                    video_suffix = f".{uploaded_video.name.split('.')[-1]}"
                    video_path = save_uploaded_file(uploaded_video, suffix=video_suffix)
                    annotated_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    annotated_video_path, zone_intervals = annotate_video(
                        video_path, annotated_video_path, helmet_conf, vest_conf,
                        helmet_model, vest_model, danger_zone_relative
                    )

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

                    if danger_zone_relative is not None:
                        st.subheader("⚠️ Tehlikeli Bölge İhlal Günlüğü")
                        if len(zone_intervals) == 0:
                            st.info("Video boyunca tehlikeli bölgeye giriş tespit edilmedi.")
                        else:
                            for i, (start, end) in enumerate(zone_intervals, start=1):
                                duration = end - start
                                st.write(
                                    f"**İhlal {i}:** {start:.1f}. saniyede girildi, "
                                    f"{end:.1f}. saniyede çıkıldı (süre: {duration:.1f} saniye)"
                                )
                except Exception as exc:
                    st.error(f"Video analizinde hata oluştu: {exc}")

with camera_tab:
    st.subheader("Kamera Analizi")
    run_webcam = st.checkbox("Kamerayı başlat", key="webcam_toggle")
    frame_placeholder = st.empty()
    warning_placeholder = st.empty()
    summary_placeholder = st.empty()

    if run_webcam:
        danger_zone_relative = st.session_state.get("danger_zone_relative")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Kameraya erişilemedi.")
        else:
            while st.session_state.get("webcam_toggle", False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Kamera görüntüsü alınamadı.")
                    break

                frame_height, frame_width = frame.shape[:2]
                danger_zone = get_absolute_zone(danger_zone_relative, frame_width, frame_height)

                annotated, summary, zone_violation = combined_predict(
                    frame, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, channels="RGB")

                if zone_violation:
                    warning_placeholder.error("⚠️ Tehlikeli bölgede kişi tespit edildi!")
                else:
                    warning_placeholder.empty()

                if len(summary) == 0:
                    summary_placeholder.info("Bu karede herhangi bir tespit yok.")
                else:
                    summary_text = "**Tespit özeti:** " + ", ".join(
                        f"{name}: {count}" for name, count in summary.items()
                    )
                    summary_placeholder.write(summary_text)

            cap.release()

with zone_tab:
    st.subheader("Tehlikeli Bölge Tanımlama")
    st.write(
        "Bir referans fotoğraf yükleyin, ardından kaydırıcılarla tehlikeli bölgenin "
        "sol-üst ve sağ-alt köşe koordinatlarını belirleyin. Bölge, önizlemede kırmızı "
        "dikdörtgen olarak anında güncellenir."
    )

    zone_reference_file = st.file_uploader(
        "Referans fotoğraf seç", type=["jpg", "jpeg", "png"], key="zone_reference"
    )

    if zone_reference_file is not None:
        zone_image = Image.open(zone_reference_file).convert("RGB")
        img_width, img_height = zone_image.size
        st.caption(f"Fotoğraf boyutu: {img_width} x {img_height} piksel")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sol-üst köşe**")
            x1 = st.slider("X1", 0, img_width, int(img_width * 0.25), key="zone_x1")
            y1 = st.slider("Y1", 0, img_height, int(img_height * 0.25), key="zone_y1")
        with col2:
            st.write("**Sağ-alt köşe**")
            x2 = st.slider("X2", 0, img_width, int(img_width * 0.75), key="zone_x2")
            y2 = st.slider("Y2", 0, img_height, int(img_height * 0.75), key="zone_y2")

        # Koordinatların mantıklı bir sırada olduğundan emin ol (x1<x2, y1<y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Canlı önizleme: fotoğrafın üzerine yarı saydam kırmızı dikdörtgen çiz
        preview = np.array(zone_image).copy()
        overlay = preview.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
        preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 3)

        st.image(preview, caption="Önizleme", use_container_width=True)

        if st.button("Bu Bölgeyi Tehlikeli Bölge Olarak Kaydet"):
            # Koordinatları MUTLAK piksel yerine ORANSAL (0-1 arası) olarak kaydediyoruz.
            # Böylece video/kamera farklı çözünürlükte olsa bile bölge doğru orana göre hesaplanır.
            relative_zone = (x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)
            st.session_state["danger_zone_relative"] = relative_zone
            st.success(
                f"Tehlikeli bölge kaydedildi (oransal): "
                f"({relative_zone[0]:.2f}, {relative_zone[1]:.2f}) - ({relative_zone[2]:.2f}, {relative_zone[3]:.2f})"
            )

    if "danger_zone_relative" in st.session_state:
        rz = st.session_state["danger_zone_relative"]
        st.info(f"Şu an kayıtlı tehlikeli bölge (oransal): {tuple(round(v, 3) for v in rz)}")
        if st.button("Tehlikeli Bölgeyi Sil"):
            del st.session_state["danger_zone_relative"]
            st.rerun()