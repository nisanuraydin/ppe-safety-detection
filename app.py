import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
from datetime import datetime
import time
from streamlit_image_coordinates import streamlit_image_coordinates

HELMET_MODEL_PATH = "runs/detect/train/weights/best.pt"
VEST_MODEL_PATH = "runs/detect/train-2/weights/best.pt"

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
    """
    if "violation_log" not in st.session_state:
        st.session_state["violation_log"] = []
    if "last_snapshot_time" not in st.session_state:
        st.session_state["last_snapshot_time"] = {}

    now = datetime.now()
    last_time = st.session_state["last_snapshot_time"].get(violation_type)

    if last_time is not None and (now - last_time).total_seconds() < SNAPSHOT_COOLDOWN_SECONDS:
        return  # çok yakın zamanda zaten kaydedildi, atla

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


def combined_predict(image, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone=None, debug_centers=None):
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

        # Süre eşiğinin altında kalan (anlık/yanlış alarm olabilecek) ihlalleri ele.
        # Kalanları (start, end, süre, kritik_mi) formatında döndür.
        filtered_intervals = []
        for start, end in zone_intervals:
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
image_tab, video_tab, camera_tab, zone_tab = st.tabs(["Fotoğraf", "Video", "Kamera", "Tehlikeli Bölge"])

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

    live_min_violation_seconds = st.slider(
        "Kritik ihlal için minimum bölgede kalma süresi (saniye)",
        0.5, 10.0, 2.0, 0.5,
        help="Kişi bölgede bu süreden az kalırsa yanlış alarm sayılır, kritik olarak işaretlenmez.",
        key="live_min_violation_seconds",
    )

    source_choice = st.radio(
        "Kaynak seç",
        ("Gerçek Webcam", "Video Dosyasını Canlı Simüle Et"),
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

    run_webcam = st.checkbox("Başlat", key="webcam_toggle")
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
                    else:
                        st.error("Kamera görüntüsü alınamadı.")
                        break

                frame = resize_frame_max_dim(frame)  # büyük fotoğraf/videolarda kutu/yazı orantısız oluyordu
                frame_height, frame_width = frame.shape[:2]
                danger_zone = get_absolute_zone(danger_zone_relative, frame_width, frame_height)

                annotated, summary, zone_violation = combined_predict(
                    frame, helmet_conf, vest_conf, helmet_model, vest_model, danger_zone
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, channels="RGB")

                ZONE_GRACE_PERIOD = 1.5  # bu süre içinde tespit kaçırılırsa sayaç sıfırlanmaz

                if zone_violation:
                    # Bölgede ne zamandır olduğunu takip et
                    if st.session_state.get("zone_entry_time") is None:
                        st.session_state["zone_entry_time"] = time.time()

                    st.session_state["zone_last_seen_time"] = time.time()
                    time_in_zone = time.time() - st.session_state["zone_entry_time"]

                    if time_in_zone >= live_min_violation_seconds:
                        warning_placeholder.error(
                            f"🔴 KRİTİK İHLAL: Kişi tehlikeli bölgede {time_in_zone:.1f} saniyedir bulunuyor!"
                        )
                        log_violation("Kritik Tehlikeli Bölge İhlali", annotated)
                    else:
                        warning_placeholder.warning(
                            f"⚠️ Kişi tehlikeli bölgede ({time_in_zone:.1f} sn, {live_min_violation_seconds:.1f} sn sonra kritik sayılacak)"
                        )
                else:
                    last_seen = st.session_state.get("zone_last_seen_time")
                    if last_seen is not None and (time.time() - last_seen) < ZONE_GRACE_PERIOD:
                        # Tespit büyük ihtimalle anlık kaçtı (flicker), sayacı KORU
                        entry_time = st.session_state.get("zone_entry_time")
                        if entry_time is not None:
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