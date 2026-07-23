import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
import torch
from datetime import datetime
import time
from io import BytesIO
from urllib.parse import urlparse
import pandas as pd
from yt_dlp import YoutubeDL
from streamlit_image_coordinates import streamlit_image_coordinates
from app_helpers import get_absolute_zone, point_in_zone

HELMET_MODEL_PATH = "runs/detect/train/weights/best.pt"
VEST_MODEL_PATH = "runs/detect/train-2/weights/best.pt"
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
INFERENCE_DEVICE_LABEL = "NVIDIA GPU" if torch.cuda.is_available() else "CPU"

st.set_page_config(page_title="İş Güvenliği Görüntü Analizi", page_icon="🦺", layout="wide")
st.title("🦺 İş Güvenliği Görüntü Analizi")
st.markdown(
    "Bu uygulama, iş güvenliği görsellerini ve videolarını analiz ederek koruyucu ekipman kullanımını tespit etmeye yardımcı olur."
)


VIOLATIONS_DIR = "violations"
SNAPSHOT_COOLDOWN_SECONDS = 5  # aynı ihlal türü için art arda kayıt sıklığını sınırlar


def ensure_violations_dir():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)


def log_violation(violation_type, annotated_frame_bgr):
    """
    Bir ihlali günlüğe ekler ve o anın görüntüsünü diske kaydeder.
    Aynı ihlal türü için son SNAPSHOT_COOLDOWN_SECONDS saniye içinde
    zaten bir kayıt yapıldıysa, tekrar kaydetmez (spam'i önler).
    Ayarlar sayfasından günlük/snapshot kapatılmışsa hiçbir şey yapmaz.
    """
    if not st.session_state.get("settings_log_enabled", True):
        return  # Ayarlardan günlük kapatılmış

    if "violation_log" not in st.session_state:
        st.session_state["violation_log"] = []
    if "last_snapshot_time" not in st.session_state:
        st.session_state["last_snapshot_time"] = {}

    now = datetime.now()
    last_time = st.session_state["last_snapshot_time"].get(violation_type)

    if last_time is not None and (now - last_time).total_seconds() < SNAPSHOT_COOLDOWN_SECONDS:
        return  # çok yakın zamanda zaten kaydedildi, atla

    filepath = None
    if st.session_state.get("settings_snapshot_enabled", True):
        ensure_violations_dir()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        safe_type = violation_type.replace(" ", "_")
        filename = f"{timestamp_str}_{safe_type}.jpg"
        filepath = os.path.join(VIOLATIONS_DIR, filename)
        cv2.imwrite(filepath, annotated_frame_bgr)

    st.session_state["violation_log"].append({
        "zaman": now.strftime("%Y-%m-%d %H:%M:%S"),
        "tur": violation_type,
        "dosya": filepath,
    })
    st.session_state["last_snapshot_time"][violation_type] = now


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


def box_center_inside(inner_box, outer_box):
    """inner_box'ın merkezi, outer_box'ın içinde mi kontrol eder."""
    ix1, iy1, ix2, iy2 = inner_box
    cx, cy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


MAX_DIMENSION = 1280  # işlenecek/gösterilecek görüntülerin en uzun kenarı bu değeri geçmeyecek


def resize_image_max_dim(pil_image, max_dim=MAX_DIMENSION):
    """PIL görüntüsünü, oranını koruyarak en uzun kenarı max_dim'i geçmeyecek şekilde küçültür."""
    w, h = pil_image.size
    if max(w, h) <= max_dim:
        return pil_image
    scale = max_dim / max(w, h)
    return pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def resize_frame_max_dim(frame, max_dim=MAX_DIMENSION):
    """Bir video karesini (numpy array), oranını koruyarak küçültür."""
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame
    scale = max_dim / max(w, h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def zone_check_enabled():
    return st.session_state.get("settings_zone_enabled", True)


def is_supported_stream_url(stream_url):
    """Doğrudan erişilebilen HLS/RTSP/video veya YouTube URL'si mi kontrol eder."""
    parsed = urlparse(stream_url.strip())
    if parsed.scheme not in {"http", "https", "rtsp", "rtsps"} or not parsed.netloc:
        return False
    return True


def is_youtube_url(stream_url):
    """URL'nin YouTube video ya da canlı yayın sayfası olup olmadığını kontrol eder."""
    hostname = (urlparse(stream_url.strip()).hostname or "").lower()
    return hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


def resolve_online_video_source(stream_url):
    """YouTube bağlantısını OpenCV'nin okuyabileceği video akışına dönüştürür."""
    if not is_supported_stream_url(stream_url):
        raise ValueError("Geçerli bir http(s), rtsp veya rtsps yayın URL'si girin.")

    if not is_youtube_url(stream_url):
        return stream_url, "doğrudan yayın"

    ydl_options = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 15,
    }
    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(stream_url, download=False)

    video_url = info.get("url")
    if not video_url:
        raise RuntimeError("YouTube yayını için oynatılabilir video akışı bulunamadı.")
    return video_url, "YouTube"


def release_live_capture():
    """Açık canlı kaynak varsa serbest bırakır."""
    capture = st.session_state.pop("live_capture", None)
    if capture is not None:
        capture.release()


def open_live_capture():
    """Aktif canlı kaynak için düşük tamponlu bir OpenCV yakalayıcısı açar."""
    config = st.session_state["live_stream_config"]
    now = time.time()

    if config.get("is_youtube") and (
        not config.get("resolved_source") or now - config.get("resolved_at", 0) > 900
    ):
        resolved_source, source_label = resolve_online_video_source(config["original_source"])
        config["resolved_source"] = resolved_source
        config["source_label"] = source_label
        config["resolved_at"] = now

    source = config.get("resolved_source", config["original_source"])
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    st.session_state["live_capture"] = capture
    return capture


def schedule_live_reconnect(reason):
    """Bağlantı hatasında artan bekleme süresiyle yeniden denemeyi planlar."""
    release_live_capture()
    config = st.session_state["live_stream_config"]
    failures = config.get("failure_count", 0) + 1
    config["failure_count"] = failures
    config["next_retry_at"] = time.time() + min(2 ** failures, 30)
    config["last_error"] = reason
    if config.get("is_youtube"):
        # YouTube'un imzalı akış URL'si süreli olabilir; sonraki denemede yenilenir.
        config["resolved_source"] = None


def combined_predict(image, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone=None, debug_centers=None):
    """
    İki modeli aynı görüntüde, KENDİ optimal güven eşikleriyle çalıştırır:
    - vest_model: boots, gloves, helmet, human, vest tespit eder (temel görsel bundan gelir)
    - helmet_model: sadece 'head' (baretsiz kafa = ihlal) bilgisini ekler, kırmızı kutu ile
    - danger_zone verilmişse: 'human' kutularının merkezi bölge içindeyse turuncu kutu + uyarı ekler

    Böylece 'helmet' sınıfı iki modelde de olduğu için tekrar/çakışma olmaz;
    helmet_model'den yalnızca ihlal (head) bilgisi alınır.
    """
    helmet_result = helmet_model.predict(image, conf=helmet_conf, verbose=False, device=INFERENCE_DEVICE)[0]
    vest_result = vest_model.predict(image, conf=vest_conf, verbose=False, device=INFERENCE_DEVICE)[0]

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

    # --- Aşama 5: Kapsamlı PPE renk kodlama ---
    # Her 'human' için: kutusunun içinde bir helmet VE bir vest var mı kontrol et.
    # İkisi de varsa yeşil ("PPE OK"), biri/ikisi eksikse kırmızı (eksik olan yazılır).
    human_class_id = None
    helmet_class_id_in_vest_model = None
    vest_class_id = None
    for idx, name in vest_model.names.items():
        if name == "human":
            human_class_id = idx
        elif name == "helmet":
            helmet_class_id_in_vest_model = idx
        elif name == "vest":
            vest_class_id = idx

    helmet_boxes_xyxy = [
        tuple(map(int, box.xyxy[0])) for box in vest_result.boxes
        if int(box.cls) == helmet_class_id_in_vest_model
    ] if helmet_class_id_in_vest_model is not None else []

    vest_boxes_xyxy = [
        tuple(map(int, box.xyxy[0])) for box in vest_result.boxes
        if int(box.cls) == vest_class_id
    ] if vest_class_id is not None else []

    ppe_ok_count = 0
    ppe_violation_count = 0

    if human_class_id is not None:
        for box in vest_result.boxes:
            if int(box.cls) != human_class_id:
                continue

            human_box = tuple(map(int, box.xyxy[0]))
            hx1, hy1, hx2, hy2 = human_box

            has_helmet = any(box_center_inside(hb, human_box) for hb in helmet_boxes_xyxy)
            has_vest = any(box_center_inside(vb, human_box) for vb in vest_boxes_xyxy)

            missing = []
            if not has_helmet:
                missing.append("No-Helmet")
            if not has_vest:
                missing.append("No-Vest")

            if missing:
                ppe_violation_count += 1
                color = (0, 0, 255)  # kırmızı (BGR)
                label = " & ".join(missing)
            else:
                ppe_ok_count += 1
                color = (0, 200, 0)  # yeşil (BGR)
                label = "PPE OK"

            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 3)
            cv2.putText(annotated, label, (hx1, min(hy2 + 20, annotated.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if ppe_ok_count > 0:
        summary["✅ PPE Uygun"] = ppe_ok_count
    if ppe_violation_count > 0:
        summary["❌ PPE İhlali"] = ppe_violation_count

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
                    # Vücudun ortası yerine AYAK noktasını (kutunun alt-orta noktası) kullanıyoruz.
                    # Çünkü tehlikeli bölge yerdeki bir alanı temsil ediyor; kişinin nerede
                    # "durduğu", vücudunun ortasından değil ayaklarının bastığı yerden anlaşılır.
                    cx, cy = (x1 + x2) // 2, y2
                    if debug_centers is not None:
                        debug_centers.append((cx, cy))
                    # Kontrol noktasını (ayak) görselde küçük bir daire ile işaretle
                    cv2.circle(annotated, (cx, cy), 6, (255, 255, 0), -1)
                    if point_in_zone(cx, cy, danger_zone):
                        zone_violation = True
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 3)
                        cv2.putText(annotated, "DANGER ZONE VIOLATION", (x1, max(y1 - 8, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                        summary["danger_zone_violation"] = summary.get("danger_zone_violation", 0) + 1

    return annotated, summary, zone_violation


def annotate_video(input_path, output_path, helmet_conf, vest_conf, helmet_model, vest_model,
                    danger_zone_relative=None, min_violation_seconds=2.0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı. Lütfen desteklenen bir video dosyası yükleyin.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_width <= 0 or orig_height <= 0:
        cap.release()
        raise RuntimeError("Video çözünürlüğü okunamadı.")

    # Çok yüksek çözünürlüklü videolarda (örn. telefon videoları) kutu/yazı boyutları
    # orantısız büyük çıkıyordu; çıktı boyutunu makul bir sınıra küçültüyoruz.
    scale = min(1.0, MAX_DIMENSION / max(orig_width, orig_height))
    width = int(orig_width * scale)
    height = int(orig_height * scale)

    # Oransal bölgeyi, bu videonun (küçültülmüş) çözünürlüğüne göre piksel koordinatına çevir
    danger_zone = get_absolute_zone(danger_zone_relative, width, height)
    if not zone_check_enabled():
        danger_zone = None

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

            if scale < 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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

        # Model tespiti bazen tek bir karede "kaçırabilir" (flicker) — bu da tek bir
        # uzun ihlali, aralarında kısa boşluklar olan birden fazla küçük parçaya bölebilir.
        # Aralarında ZONE_GRACE_PERIOD'dan az boşluk olan aralıkları BİRLEŞTİRİYORUZ.
        ZONE_GRACE_PERIOD = 1.5
        merged_intervals = []
        for start, end in zone_intervals:
            if merged_intervals and (start - merged_intervals[-1][1]) < ZONE_GRACE_PERIOD:
                # Önceki aralığa çok yakın, birleştir (bitişini uzat)
                merged_intervals[-1] = (merged_intervals[-1][0], end)
            else:
                merged_intervals.append((start, end))

        # Süre eşiğinin altında kalan (anlık/yanlış alarm olabilecek) ihlalleri ele.
        # Kalanları (start, end, süre, kritik_mi) formatında döndür.
        filtered_intervals = []
        for start, end in merged_intervals:
            duration = end - start
            if duration >= min_violation_seconds:
                filtered_intervals.append((start, end, duration, True))
            # Eşiğin altındakiler günlüğe hiç eklenmiyor (yanlış alarm sayılıyor)

        return output_path, filtered_intervals
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
image_tab, video_tab, camera_tab, zone_tab, dashboard_tab, settings_tab = st.tabs(
    ["Fotoğraf", "Video", "Kamera", "Tehlikeli Bölge", "📊 Dashboard", "⚙️ Ayarlar"]
)

with image_tab:
    st.subheader("Fotoğraf Analizi")
    uploaded_file = st.file_uploader("Fotoğraf seç", type=["jpg", "jpeg", "png"], key="image_input")
    if uploaded_file is not None:
        if st.button("Analiz Et", key="analyze_image"):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                image = resize_image_max_dim(image)  # çok büyük fotoğraflarda kutu/yazı boyutu orantısız oluyordu
                with st.spinner("Görsel analiz ediliyor..."):
                    relative_zone = st.session_state.get("danger_zone_relative")
                    danger_zone = get_absolute_zone(relative_zone, image.width, image.height)
                    if not zone_check_enabled():
                        danger_zone = None
                    annotated_bgr, summary, zone_violation = combined_predict(
                        image, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
                    )
                    annotated = annotated_bgr[:, :, ::-1]
                    st.session_state["last_summary"] = summary
                    st.session_state["last_summary_source"] = "Fotoğraf"
                    st.image(annotated, caption="Tespit sonucu", width=700)

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
    min_violation_seconds = st.slider(
        "Kritik ihlal için minimum bölgede kalma süresi (saniye)",
        0.5, 10.0, 2.0, 0.5,
        help="Bu süreden kısa süren bölge girişleri, yanlış alarm sayılıp günlüğe eklenmez.",
        key="min_violation_seconds",
    )
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
                        helmet_model, vest_model, danger_zone_relative, min_violation_seconds
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
                            for i, (start, end, duration, is_critical) in enumerate(zone_intervals, start=1):
                                critical_tag = "🔴 **KRİTİK**" if is_critical else ""
                                st.write(
                                    f"**İhlal {i}:** {start:.1f}. saniyede girildi, "
                                    f"{end:.1f}. saniyede çıkıldı (süre: {duration:.1f} saniye) {critical_tag}"
                                )
                except Exception as exc:
                    st.error(f"Video analizinde hata oluştu: {exc}")

with camera_tab:
    st.subheader("Kamera Analizi")
    st.caption(f"İşlem birimi: {INFERENCE_DEVICE_LABEL}")

    live_min_violation_seconds = st.slider(
        "Kritik ihlal için minimum bölgede kalma süresi (saniye)",
        0.5, 10.0, 2.0, 0.5,
        help="Kişi bölgede bu süreden az kalırsa yanlış alarm sayılır, kritik olarak işaretlenmez.",
        key="live_min_violation_seconds",
    )

    source_choice = st.radio(
        "Kaynak seç",
        ("Gerçek Webcam", "Video Dosyasını Canlı Simüle Et", "Online Video / Yayın URL'si"),
        key="camera_source_choice",
    )

    simulated_video_path = None
    if source_choice == "Video Dosyasını Canlı Simüle Et":
        sim_video_file = st.file_uploader(
            "Simüle edilecek videoyu seç", type=["mp4", "mov", "avi", "mkv", "webm"], key="sim_video_input"
        )
        if sim_video_file is not None:
            simulated_video_path = save_uploaded_file(sim_video_file, suffix=f".{sim_video_file.name.split('.')[-1]}")
            st.caption("Video, gerçek zamanına yakın bir hızda, canlı yayınmış gibi oynatılacak.")

    online_stream_url = None
    if source_choice == "Online Video / Yayın URL'si":
        online_stream_url = st.text_input(
            "Video veya yayın URL'si",
            placeholder="YouTube bağlantısı, https://ornek.com/yayin.m3u8 veya rtsp://kamera-adresi/stream",
            help="YouTube video/canlı yayın bağlantısı ya da doğrudan HLS (.m3u8), MP4, RTSP/RSTPS yayın adresi girebilirsiniz.",
            key="online_stream_url",
        ).strip()
        st.caption("Yalnızca kullanım izniniz olan veya herkese açık yayınları kullanın. YouTube akışlarının kullanılabilirliği yayın sahibine bağlıdır.")

    if INFERENCE_DEVICE == 0:
        target_analysis_fps = st.slider(
            "Hedef analiz hızı (kare/sn)", 1, 20, 10,
            help="RTX 4060 ile iki model GPU'da çalışıyor. Daha yüksek değer daha akıcı güncelleme sağlar.",
            key="live_target_analysis_fps_gpu",
        )
    else:
        target_analysis_fps = st.slider(
            "Hedef analiz hızı (kare/sn)", 0.25, 2.0, 0.5, 0.25,
            help="CPU kullanımında düşük değer gecikmeyi ve işlem yükünü azaltır.",
            key="live_target_analysis_fps_cpu",
        )
    start_col, stop_col, status_col = st.columns([1, 1, 3])
    start_clicked = start_col.button(
        "Canlı Analizi Başlat", type="primary", disabled=st.session_state.get("live_stream_active", False)
    )
    stop_clicked = stop_col.button("Durdur", disabled=not st.session_state.get("live_stream_active", False))

    if stop_clicked:
        st.session_state["live_stream_active"] = False
        release_live_capture()
        st.rerun()

    if start_clicked:
        try:
            if source_choice == "Video Dosyasını Canlı Simüle Et":
                if simulated_video_path is None:
                    raise ValueError("Önce simüle edilecek bir video yükle.")
                source, source_label, is_youtube = simulated_video_path, "video simülasyonu", False
            elif source_choice == "Online Video / Yayın URL'si":
                if not online_stream_url:
                    raise ValueError("Önce bir video veya yayın URL'si girin.")
                source, source_label = resolve_online_video_source(online_stream_url)
                is_youtube = is_youtube_url(online_stream_url)
            else:
                source, source_label, is_youtube = 0, "yerel webcam", False

            release_live_capture()
            st.session_state["live_stream_config"] = {
                "original_source": online_stream_url if source_choice == "Online Video / Yayın URL'si" else source,
                "resolved_source": source,
                "source_label": source_label,
                "is_youtube": is_youtube,
                "resolved_at": time.time(),
                "failure_count": 0,
                "next_retry_at": 0,
                "last_error": None,
                "source_choice": source_choice,
            }
            st.session_state["live_stream_active"] = True
            st.session_state["zone_entry_time"] = None
            st.session_state["zone_last_seen_time"] = None
            st.session_state["live_analysis_fps"] = None
            st.rerun()
        except Exception as exc:
            st.error(f"Canlı kaynak başlatılamadı: {exc}")

    if st.session_state.get("live_stream_active", False):
        status_col.success(f"Aktif kaynak: {st.session_state['live_stream_config']['source_label']}")
    else:
        status_col.info("Canlı analiz beklemede.")

    @st.fragment(run_every=1 / target_analysis_fps)
    def render_live_stream():
        if not st.session_state.get("live_stream_active", False):
            return

        config = st.session_state["live_stream_config"]
        now = time.time()
        if now < config.get("next_retry_at", 0):
            retry_in = config["next_retry_at"] - now
            st.warning(f"Bağlantı yeniden deneniyor ({retry_in:.0f} sn): {config.get('last_error', '')}")
            return

        try:
            capture = st.session_state.get("live_capture")
            if capture is None or not capture.isOpened():
                capture = open_live_capture()
                if not capture.isOpened():
                    schedule_live_reconnect("Kaynak açılamadı")
                    st.warning("Kaynağa erişilemedi; yeniden denenecek.")
                    return

            ret, frame = capture.read()
            if not ret and config["source_choice"] == "Video Dosyasını Canlı Simüle Et":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = capture.read()
            if not ret:
                schedule_live_reconnect("Yayından kare alınamadı")
                st.warning("Kare alınamadı; bağlantı yeniden kurulacak.")
                return

            # Tamponda bekleyen eski kareleri atarak gecikmenin büyümesini önle.
            for _ in range(2):
                if not capture.grab():
                    break
                fresh_ret, fresh_frame = capture.retrieve()
                if fresh_ret:
                    frame = fresh_frame

            started_at = time.time()
            frame = resize_frame_max_dim(frame)
            frame_height, frame_width = frame.shape[:2]
            danger_zone = get_absolute_zone(st.session_state.get("danger_zone_relative"), frame_width, frame_height)
            if not zone_check_enabled():
                danger_zone = None
            annotated, summary, zone_violation = combined_predict(
                frame, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
            )

            processing_seconds = time.time() - started_at
            instant_fps = 1 / processing_seconds if processing_seconds else 0
            previous_fps = st.session_state.get("live_analysis_fps")
            st.session_state["live_analysis_fps"] = instant_fps if previous_fps is None else 0.7 * previous_fps + 0.3 * instant_fps
            config["failure_count"] = 0
            config["last_error"] = None
            st.session_state["last_summary"] = summary
            st.session_state["last_summary_source"] = f"Kamera ({config['source_label']})"

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            st.caption(f"Analiz hızı: {st.session_state['live_analysis_fps']:.1f} FPS | Hedef: {target_analysis_fps} FPS")

            zone_grace_period = 1.5
            alarm_enabled = st.session_state.get("settings_alarm_enabled", True)
            if zone_violation:
                if st.session_state.get("zone_entry_time") is None:
                    st.session_state["zone_entry_time"] = time.time()
                st.session_state["zone_last_seen_time"] = time.time()
                time_in_zone = time.time() - st.session_state["zone_entry_time"]
                if time_in_zone >= live_min_violation_seconds:
                    if alarm_enabled:
                        st.error(f"🔴 KRİTİK İHLAL: Kişi tehlikeli bölgede {time_in_zone:.1f} saniyedir bulunuyor!")
                    log_violation("Kritik Tehlikeli Bölge İhlali", annotated)
                elif alarm_enabled:
                    st.warning(f"⚠️ Kişi tehlikeli bölgede ({time_in_zone:.1f} sn, {live_min_violation_seconds:.1f} sn sonra kritik sayılacak)")
            else:
                last_seen = st.session_state.get("zone_last_seen_time")
                if last_seen is None or time.time() - last_seen >= zone_grace_period:
                    st.session_state["zone_entry_time"] = None
                    st.session_state["zone_last_seen_time"] = None

            if summary.get("head (baretsiz)", 0) > 0:
                log_violation("Baretsiz (No-Helmet)", annotated)
            st.write("**Tespit özeti:** " + (", ".join(f"{name}: {count}" for name, count in summary.items()) if summary else "Bu karede tespit yok."))
        except Exception as exc:
            schedule_live_reconnect(str(exc))
            st.warning(f"Canlı akış hatası; yeniden denenecek: {exc}")

    if st.session_state.get("live_stream_active", False):
        render_live_stream()

    # Önceki blok tek bir while döngüsüyle Streamlit arayüzünü kilitliyordu.
    run_webcam = False
    frame_placeholder = st.empty()
    warning_placeholder = st.empty()
    summary_placeholder = st.empty()

    if run_webcam:
        danger_zone_relative = st.session_state.get("danger_zone_relative")

        if source_choice == "Video Dosyasını Canlı Simüle Et":
            if simulated_video_path is None:
                st.warning("Önce simüle edilecek bir video yükle.")
                st.stop()
            video_source = simulated_video_path
        elif source_choice == "Online Video / Yayın URL'si":
            if not online_stream_url:
                st.warning("Önce bir video veya yayın URL'si girin.")
                st.stop()
            try:
                video_source, source_label = resolve_online_video_source(online_stream_url)
                st.caption(f"Kaynak hazırlandı: {source_label}")
            except Exception as exc:
                st.error(f"Online kaynak hazırlanamadı: {exc}")
                st.stop()
        else:
            video_source = 0

        st.session_state["zone_entry_time"] = None  # her yeni başlatmada sayaç sıfırlanır
        st.session_state["zone_last_seen_time"] = None
        cap = cv2.VideoCapture(video_source)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / source_fps  # gerçek videonun hızına yakın oynatmak için

        if not cap.isOpened():
            st.error("Kaynağa erişilemedi.")
        else:
            while st.session_state.get("webcam_toggle", False):
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    if source_choice == "Video Dosyasını Canlı Simüle Et":
                        # Video bittiğinde başa sar, canlı yayın hissi sürsün
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    elif source_choice == "Online Video / Yayın URL'si":
                        st.error("Online yayından kare alınamadı. Yayın adresini ve erişim iznini kontrol edin.")
                        break
                    else:
                        st.error("Kamera görüntüsü alınamadı.")
                        break

                frame = resize_frame_max_dim(frame)  # büyük fotoğraf/videolarda kutu/yazı orantısız oluyordu
                frame_height, frame_width = frame.shape[:2]
                danger_zone = get_absolute_zone(danger_zone_relative, frame_width, frame_height)
                if not zone_check_enabled():
                    danger_zone = None

                annotated, summary, zone_violation = combined_predict(
                    frame, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
                )
                st.session_state["last_summary"] = summary
                st.session_state["last_summary_source"] = "Kamera"
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, channels="RGB")

                ZONE_GRACE_PERIOD = 1.5  # bu süre içinde tespit kaçırılırsa sayaç sıfırlanmaz

                alarm_enabled = st.session_state.get("settings_alarm_enabled", True)

                if zone_violation:
                    # Bölgede ne zamandır olduğunu takip et
                    if st.session_state.get("zone_entry_time") is None:
                        st.session_state["zone_entry_time"] = time.time()

                    st.session_state["zone_last_seen_time"] = time.time()
                    time_in_zone = time.time() - st.session_state["zone_entry_time"]

                    if time_in_zone >= live_min_violation_seconds:
                        if alarm_enabled:
                            warning_placeholder.error(
                                f"🔴 KRİTİK İHLAL: Kişi tehlikeli bölgede {time_in_zone:.1f} saniyedir bulunuyor!"
                            )
                        log_violation("Kritik Tehlikeli Bölge İhlali", annotated)
                    elif alarm_enabled:
                        warning_placeholder.warning(
                            f"⚠️ Kişi tehlikeli bölgede ({time_in_zone:.1f} sn, {live_min_violation_seconds:.1f} sn sonra kritik sayılacak)"
                        )
                else:
                    last_seen = st.session_state.get("zone_last_seen_time")
                    if last_seen is not None and (time.time() - last_seen) < ZONE_GRACE_PERIOD:
                        # Tespit büyük ihtimalle anlık kaçtı (flicker), sayacı KORU
                        entry_time = st.session_state.get("zone_entry_time")
                        if entry_time is not None and alarm_enabled:
                            time_in_zone = time.time() - entry_time
                            warning_placeholder.warning(
                                f"⚠️ Kişi tehlikeli bölgede ({time_in_zone:.1f} sn, tespit anlık kaçtı)"
                            )
                    else:
                        # Gerçekten bölgeden çıkmış / uzun süredir görünmüyor
                        st.session_state["zone_entry_time"] = None
                        st.session_state["zone_last_seen_time"] = None
                        warning_placeholder.empty()

                if summary.get("head (baretsiz)", 0) > 0:
                    log_violation("Baretsiz (No-Helmet)", annotated)

                if len(summary) == 0:
                    summary_placeholder.info("Bu karede herhangi bir tespit yok.")
                else:
                    summary_text = "**Tespit özeti:** " + ", ".join(
                        f"{name}: {count}" for name, count in summary.items()
                    )
                    summary_placeholder.write(summary_text)

                # Video simülasyonunda gerçek oynatma hızına yakın kalmak için bekle.
                # (Gerçek webcam zaten kendi hızında akar, bu satır ona ekstra yavaşlık katmaz.)
                elapsed = time.time() - loop_start
                remaining = frame_delay - elapsed
                if source_choice == "Video Dosyasını Canlı Simüle Et" and remaining > 0:
                    time.sleep(remaining)

            cap.release()

    st.divider()
    st.subheader("📋 İhlal Günlüğü")
    violation_log = st.session_state.get("violation_log", [])
    if len(violation_log) == 0:
        st.info("Henüz kaydedilmiş bir ihlal yok.")
    else:
        st.write(f"Toplam **{len(violation_log)}** ihlal kaydedildi.")
        # En yeni ihlal en üstte görünsün
        for entry in reversed(violation_log[-20:]):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{entry['zaman']}** — {entry['tur']}")
                st.caption(entry["dosya"])
            with col_b:
                if os.path.exists(entry["dosya"]):
                    st.image(entry["dosya"], width=120)

        if st.button("Günlüğü Temizle"):
            st.session_state["violation_log"] = []
            st.session_state["last_snapshot_time"] = {}
            st.rerun()

with zone_tab:
    st.subheader("Tehlikeli Bölge Tanımlama")

    zone_method = st.radio(
        "Yöntem seç",
        ("Fare ile Çiz (Yeni)", "Sayısal Kaydırıcılar (Yedek)"),
        key="zone_method",
        horizontal=True,
    )

    zone_reference_file = st.file_uploader(
        "Referans fotoğraf seç", type=["jpg", "jpeg", "png"], key="zone_reference"
    )

    if zone_reference_file is not None:
        zone_image = Image.open(zone_reference_file).convert("RGB")
        img_width, img_height = zone_image.size
        st.caption(f"Fotoğraf boyutu: {img_width} x {img_height} piksel")

        if zone_method == "Fare ile Çiz (Yeni)":
            st.write("Fotoğrafın üzerine tıkla, basılı tut, sürükle ve bırak — dikdörtgen çizilecek.")
            st.caption("Çizim bittikten sonra aşağıda kırmızı önizleme belirecek.")

            DISPLAY_WIDTH = 700  # sayfa oranının bozulmaması için görüntüyü bu genişliğe sınırlıyoruz

            coords = streamlit_image_coordinates(
                zone_image, key="zone_draw", click_and_drag=True, width=DISPLAY_WIDTH
            )

            if coords is not None:
                # Bileşenin kendi 'width'/'height' alanlarını kullanıyoruz — bu, bileşenin
                # görüntüyü GERÇEKTE hangi boyutta çizdiğini gösteriyor. Orijinal fotoğraf
                # boyutuyla aynı olmayabilir, o yüzden oranı buna göre hesaplamak daha güvenli.
                draw_width = coords["width"]
                draw_height = coords["height"]
                x1, y1 = int(coords["x1"]), int(coords["y1"])
                x2, y2 = int(coords["x2"]), int(coords["y2"])
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                relative_zone = (x1 / draw_width, y1 / draw_height, x2 / draw_width, y2 / draw_height)

                st.caption(
                    f"🔍 Debug — Çizim alanı boyutu: {draw_width}x{draw_height} | "
                    f"Seçilen piksel: ({x1},{y1})-({x2},{y2}) | "
                    f"Orijinal fotoğraf: {img_width}x{img_height} | "
                    f"Oransal: {tuple(round(v, 3) for v in relative_zone)}"
                )

                # Çizilen alanı, orijinal fotoğrafın üzerine (oranı koruyarak) çizip önizleme gösteriyoruz
                preview = np.array(zone_image).copy()
                px1 = int(relative_zone[0] * img_width)
                py1 = int(relative_zone[1] * img_height)
                px2 = int(relative_zone[2] * img_width)
                py2 = int(relative_zone[3] * img_height)
                overlay = preview.copy()
                cv2.rectangle(overlay, (px1, py1), (px2, py2), (255, 0, 0), -1)
                preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)
                cv2.rectangle(preview, (px1, py1), (px2, py2), (255, 0, 0), 3)

                # Önizlemeyi sabit, makul bir genişliğe küçült (konteyner genişliğine göre değil)
                # böylece dikey/portre fotoğraflarda dev bir kutu oluşmasın.
                preview_pil = Image.fromarray(preview)
                preview_pil.thumbnail((500, 500))
                st.image(preview_pil, caption="Önizleme (kaydedilecek alan)")

                if st.button("Bu Bölgeyi Tehlikeli Bölge Olarak Kaydet", key="save_zone_draw"):
                    st.session_state["danger_zone_relative"] = relative_zone
                    st.success(
                        f"Tehlikeli bölge kaydedildi (oransal): "
                        f"({relative_zone[0]:.2f}, {relative_zone[1]:.2f}) - "
                        f"({relative_zone[2]:.2f}, {relative_zone[3]:.2f})"
                    )
            else:
                st.info("Henüz bir alan çizilmedi.")

        else:  # Sayısal Kaydırıcılar (Yedek)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sol-üst köşe**")
                x1 = st.slider("X1", 0, img_width, int(img_width * 0.25), key="zone_x1")
                y1 = st.slider("Y1", 0, img_height, int(img_height * 0.25), key="zone_y1")
            with col2:
                st.write("**Sağ-alt köşe**")
                x2 = st.slider("X2", 0, img_width, int(img_width * 0.75), key="zone_x2")
                y2 = st.slider("Y2", 0, img_height, int(img_height * 0.75), key="zone_y2")

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            preview = np.array(zone_image).copy()
            overlay = preview.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
            preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 3)

            st.image(preview, caption="Önizleme", use_container_width=True)

            if st.button("Bu Bölgeyi Tehlikeli Bölge Olarak Kaydet", key="save_zone_slider"):
                relative_zone = (x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)
                st.session_state["danger_zone_relative"] = relative_zone
                st.success(
                    f"Tehlikeli bölge kaydedildi (oransal): "
                    f"({relative_zone[0]:.2f}, {relative_zone[1]:.2f}) - "
                    f"({relative_zone[2]:.2f}, {relative_zone[3]:.2f})"
                )

    if "danger_zone_relative" in st.session_state:
        rz = st.session_state["danger_zone_relative"]
        st.info(f"Şu an kayıtlı tehlikeli bölge (oransal): {tuple(round(v, 3) for v in rz)}")
        if st.button("Tehlikeli Bölgeyi Sil"):
            del st.session_state["danger_zone_relative"]
            st.rerun()

with dashboard_tab:
    st.subheader("📊 Canlı Dashboard")

    last_summary = st.session_state.get("last_summary")
    last_source = st.session_state.get("last_summary_source", "-")

    if last_summary is None:
        st.info("Henüz bir analiz yapılmadı. Fotoğraf veya Kamera sekmesinden bir analiz çalıştır.")
    else:
        st.caption(f"Son analiz kaynağı: {last_source}")

        total_workers = last_summary.get("human", 0)
        ppe_ok = last_summary.get("✅ PPE Uygun", 0)
        ppe_violation = last_summary.get("❌ PPE İhlali", 0)
        zone_violation_count = last_summary.get("danger_zone_violation", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👷 Toplam Çalışan", total_workers)
        col2.metric("✅ PPE Uygun", ppe_ok)
        col3.metric("❌ PPE İhlali", ppe_violation)
        col4.metric("🚧 Bölge İhlali", zone_violation_count)

    st.divider()
    st.subheader("Bu Oturumdaki İhlal İstatistikleri")

    violation_log = st.session_state.get("violation_log", [])
    if len(violation_log) == 0:
        st.info("Henüz kaydedilmiş bir ihlal yok.")
    else:
        type_counts = {}
        for entry in violation_log:
            type_counts[entry["tur"]] = type_counts.get(entry["tur"], 0) + 1

        total_violations = len(violation_log)
        most_common_type, most_common_count = max(type_counts.items(), key=lambda item: item[1])
        metric_col, common_col, type_col = st.columns(3)
        metric_col.metric("Toplam Kayıt", total_violations)
        common_col.metric("En Sık İhlal", most_common_type, f"{most_common_count} kayıt")
        type_col.metric("Farklı İhlal Türü", len(type_counts))

        stats_df = pd.DataFrame(
            [
                {
                    "İhlal Türü": violation_type,
                    "Kayıt Sayısı": count,
                    "Oran": f"{count / total_violations:.0%}",
                }
                for violation_type, count in type_counts.items()
            ]
        ).sort_values("Kayıt Sayısı", ascending=False, ignore_index=True)

        table_col, chart_col = st.columns([3, 2])
        with table_col:
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        with chart_col:
            st.bar_chart(stats_df.set_index("İhlal Türü")[["Kayıt Sayısı"]], use_container_width=True)

        st.divider()
        st.subheader("📄 Otomatik Rapor Oluştur")

        report_df = pd.DataFrame(violation_log)[["zaman", "tur", "dosya"]]
        report_df.columns = ["Zaman", "İhlal Türü", "Fotoğraf Dosyası"]

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            report_df.to_excel(writer, index=False, sheet_name="İhlal Raporu")

            summary_df = pd.DataFrame(
                [{"İhlal Türü": tur, "Adet": count} for tur, count in type_counts.items()]
            )
            summary_df.to_excel(writer, index=False, sheet_name="Özet")

        st.download_button(
            "📥 Excel Raporu İndir (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"ihlal_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with settings_tab:
    st.subheader("⚙️ Sistem Ayarları")
    st.write("Bu ayarlar, tüm sekmelerdeki analiz davranışını etkiler.")

    st.toggle(
        "🚧 Tehlikeli Bölge Kontrolü Aktif",
        value=st.session_state.get("settings_zone_enabled", True),
        key="settings_zone_enabled",
        help="Kapatılırsa, tanımlı bölge olsa bile hiçbir bölge kontrolü yapılmaz.",
    )
    st.toggle(
        "🔔 Alarm / Uyarı Sistemi Aktif",
        value=st.session_state.get("settings_alarm_enabled", True),
        key="settings_alarm_enabled",
        help="Kapatılırsa, canlı kamera ekranında ihlal uyarı mesajları gösterilmez (kayıt yine de tutulabilir).",
    )
    st.toggle(
        "📸 Otomatik Snapshot Kaydı Aktif",
        value=st.session_state.get("settings_snapshot_enabled", True),
        key="settings_snapshot_enabled",
        help="Kapatılırsa, ihlal anında fotoğraf diske kaydedilmez (günlük kaydı yine tutulabilir).",
    )
    st.toggle(
        "📋 İhlal Günlüğü Aktif",
        value=st.session_state.get("settings_log_enabled", True),
        key="settings_log_enabled",
        help="Kapatılırsa, hiçbir ihlal günlüğe eklenmez ve fotoğraf kaydedilmez.",
    )

    st.divider()
    st.caption(
        "Güven eşikleri (baret/yelek modeli) sayfanın üst kısmındaki kaydırıcılardan, "
        "kritik ihlal süresi ise Video ve Kamera sekmelerindeki kaydırıcılardan ayarlanır."
    )
