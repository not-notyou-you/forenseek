"""Forenseek — prototipe deteksi deepfake wajah dengan penjelasan Grad-CAM.

Struktur yang diharapkan:
    models/model.keras        model terlatih (ekspor dari notebook bagian 9)
    models/model_meta.json    {"model_name", "threshold", "gradcam_layer", "img_size", ...}
Alternatif: isi st.secrets["MODEL_URL"] dan st.secrets["META_URL"] untuk mengunduh model saat start.
"""
from __future__ import annotations

import io
import json
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

APP_NAME = "Forenseek"
HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_PATH = MODEL_DIR / "model.keras"
META_PATH = MODEL_DIR / "model_meta.json"
DEFAULT_META = {"model_name": "xception", "threshold": 0.5, "gradcam_layer": "block14_sepconv2_act", "img_size": 224}

st.set_page_config(page_title=APP_NAME, page_icon="◐", layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- gaya
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --navy-900:#071330;--navy-800:#0B1B3F;--navy-700:#13285A;--navy-500:#2F4C8C;--accent:#3D7BFF;
  --ink:#0B1B3F;--muted:#5B6B8C;--faint:#8A97B3;--line:#E4E9F3;--bg:#F3F6FB;--card:#FFFFFF;
  --real:#14936B;--real-bg:#E7F6F0;--fake:#D14343;--fake-bg:#FCECEC;
  --sh-1:0 1px 2px rgba(11,27,63,.06),0 1px 3px rgba(11,27,63,.05);
  --sh-2:0 2px 4px rgba(11,27,63,.04),0 8px 24px -6px rgba(11,27,63,.10);
  --sh-3:0 4px 8px rgba(11,27,63,.05),0 22px 48px -12px rgba(11,27,63,.22);
  --r:18px;
}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,sans-serif;color:var(--ink);}
.stApp{background:var(--bg);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1160px;}
code{background:#EEF2FA!important;color:var(--navy-700)!important;border-radius:6px;padding:1px 6px;}

/* ---------- navbar ---------- */
.nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em;font-size:1.02rem;}
.mark{width:30px;height:30px;border-radius:9px;background:var(--navy-800);box-shadow:var(--sh-2);position:relative;}
.mark:before{content:"";position:absolute;inset:7px;border-radius:50%;border:2px solid #fff;}
.mark:after{content:"";position:absolute;left:12px;top:12px;width:6px;height:6px;border-radius:50%;background:var(--accent);}
.pill{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;
      color:var(--navy-500);background:#fff;border:2px solid var(--line);border-radius:99px;padding:5px 11px;box-shadow:var(--sh-1);}

/* ---------- hero ---------- */
.hero{background:linear-gradient(135deg,var(--navy-900) 0%,var(--navy-800) 55%,var(--navy-700) 100%);
      border-radius:24px;padding:40px 44px 34px;color:#fff;position:relative;overflow:hidden;box-shadow:var(--sh-3);margin-bottom:30px;}
.hero:before{content:"";position:absolute;right:-90px;top:-110px;width:380px;height:380px;border-radius:50%;
      border:2px solid rgba(255,255,255,.10);box-shadow:0 0 0 46px rgba(255,255,255,.025),0 0 0 92px rgba(255,255,255,.018);}
.eyebrow{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#8FA6DA;}
.hero h1{font-weight:700;font-size:2.6rem;letter-spacing:-.035em;margin:10px 0 10px;color:#fff;line-height:1.05;}
.hero p{margin:0;color:#B9C6E4;font-weight:300;font-size:1.02rem;max-width:600px;line-height:1.6;}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px;position:relative;z-index:1;}
.stat{background:rgba(255,255,255,.07);border:2px solid rgba(255,255,255,.12);border-radius:12px;padding:9px 14px;}
.stat b{display:block;font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:#8FA6DA;font-weight:500;}
.stat span{font-size:.92rem;font-weight:500;color:#fff;}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle;}

/* ---------- tabs sebagai segmented control ---------- */
[data-testid="stTabs"] [role="tablist"]{background:#fff;border:2px solid var(--line);border-radius:14px;padding:5px;gap:4px;
      box-shadow:var(--sh-1);width:fit-content;margin-bottom:22px;}
[data-testid="stTab"]{border-radius:10px!important;padding:8px 20px!important;height:auto!important;transition:all .15s;}
[data-testid="stTab"] p{font-weight:500;font-size:.92rem;color:var(--muted);}
[data-testid="stTab"]:hover p{color:var(--ink);}
[data-testid="stTab"][data-selected="true"]{background:var(--navy-800);box-shadow:var(--sh-2);}
[data-testid="stTab"][data-selected="true"] p{color:#fff;}
[data-testid="stTabs"] .react-aria-SelectionIndicator{display:none;}

/* ---------- section heading ---------- */
.step{display:flex;align-items:baseline;gap:12px;margin:6px 0 12px;}
.step .n{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--accent);font-weight:500;}
.step h3{margin:0;font-size:1.12rem;font-weight:600;letter-spacing:-.01em;color:var(--ink);}
.step small{color:var(--faint);font-size:.85rem;}

/* ---------- cards ---------- */
.card{background:var(--card);border:2px solid var(--line);border-radius:var(--r);padding:24px 26px;box-shadow:var(--sh-2);}
[class*="st-key-card_"]{background:#fff;border:2px solid var(--line);border-radius:var(--r);padding:20px 22px;box-shadow:var(--sh-2);}
.label{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-bottom:8px;}

/* hasil utama */
.result{position:relative;overflow:hidden;box-shadow:var(--sh-3);}
.result:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;}
.result.fake:before{background:var(--fake);} .result.real:before{background:var(--real);}
.chip{display:inline-flex;align-items:center;gap:7px;font-weight:600;font-size:.86rem;border-radius:99px;padding:6px 12px;}
.chip.fake{background:var(--fake-bg);color:var(--fake);} .chip.real{background:var(--real-bg);color:var(--real);}
.score{font-family:'JetBrains Mono',monospace;font-size:3.4rem;font-weight:500;color:var(--navy-800);line-height:1;margin:18px 0 4px;letter-spacing:-.03em;}
.score small{font-size:1.3rem;color:var(--faint);}
.bar{height:10px;border-radius:99px;background:#EDF1F8;overflow:visible;margin:18px 0 8px;position:relative;}
.bar>span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--navy-700),var(--accent));}
.bar>i{position:absolute;top:-5px;width:3px;height:20px;border-radius:2px;background:var(--navy-800);box-shadow:0 0 0 3px #fff;}
.scale{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--faint);}
.note{color:var(--muted);font-size:.86rem;line-height:1.6;}
.divider{height:2px;background:var(--line);margin:18px 0 14px;}

.empty{background:#fff;border:2px dashed #C9D3E8;border-radius:var(--r);padding:34px 28px;text-align:center;box-shadow:var(--sh-1);}
.empty .ic{width:44px;height:44px;margin:0 auto 12px;border-radius:12px;background:#EEF2FA;position:relative;}
.empty .ic:before{content:"";position:absolute;inset:13px;border-radius:50%;border:2px solid var(--navy-500);}
.empty h4{margin:0 0 6px;font-size:1rem;font-weight:600;color:var(--ink);}

.legend{height:8px;border-radius:99px;background:linear-gradient(90deg,#00007F,#0080FF,#7FFF7F,#FF8000,#7F0000);margin:10px 0 4px;}

/* ---------- uploader & widget ---------- */
[data-testid="stFileUploader"] section,[data-testid="stFileUploaderDropzone"]{background:#fff;border:2px dashed #BFCBE3!important;
      border-radius:16px!important;box-shadow:var(--sh-1);transition:all .15s;padding:22px!important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent)!important;box-shadow:var(--sh-2);}
[data-testid="stFileUploader"] button{background:var(--navy-800)!important;color:#fff!important;border:0!important;border-radius:10px!important;box-shadow:var(--sh-2);}
[data-testid="stImage"] img{border-radius:14px;box-shadow:var(--sh-2);}
[data-testid="stImageCaption"],[data-testid="caption"]{color:var(--muted)!important;}
.stProgress > div > div > div > div{background:var(--navy-700);}
.foot{display:flex;justify-content:space-between;color:var(--faint);font-size:.78rem;margin-top:40px;border-top:2px solid var(--line);padding-top:16px;}
@media (max-width:640px){.hero{padding:28px 22px}.hero h1{font-size:2rem}.score{font-size:2.6rem}}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- model
def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())


@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    """Kembalikan (model, meta) atau (None, meta) bila model belum tersedia."""
    try:
        if not MODEL_PATH.exists() and "MODEL_URL" in st.secrets:
            _download(st.secrets["MODEL_URL"], MODEL_PATH)
        if not META_PATH.exists() and "META_URL" in st.secrets:
            _download(st.secrets["META_URL"], META_PATH)
    except Exception as e:  # secrets tidak ada / unduhan gagal
        st.session_state["load_error"] = str(e)
    meta = {**DEFAULT_META, **(json.loads(META_PATH.read_text()) if META_PATH.exists() else {})}
    if not MODEL_PATH.exists():
        return None, meta
    import tensorflow as tf
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, meta


@st.cache_resource
def face_detector():
    try:
        clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if clf.empty() else clf
    except Exception:
        return None


def crop_face(img: Image.Image, margin: float = 0.25) -> tuple[Image.Image, bool]:
    """Potong wajah terbesar (dengan margin). Jika tidak terdeteksi, kembalikan citra utuh."""
    rgb = np.asarray(img.convert("RGB"))
    det = face_detector()
    if det is None:
        return img.convert("RGB"), False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return img.convert("RGB"), False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(max(w, h) * margin)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    return Image.fromarray(rgb[y0:y1, x0:x1]), True


def preprocess(img: Image.Image, meta: dict) -> np.ndarray:
    s = int(meta["img_size"])
    x = np.asarray(img.convert("RGB").resize((s, s), Image.BILINEAR), dtype=np.float32)
    name = meta["model_name"]
    if name == "xception":
        x = x / 127.5 - 1.0
    elif name == "mesonet":
        x = x / 255.0
    return x[None]  # efficientnet: 0-255 (rescaling di dalam model)


def predict_with_gradcam(model, meta: dict, img: Image.Image):
    """Probabilitas deepfake + heatmap Grad-CAM (terhadap log-odds kelas deepfake)."""
    import tensorflow as tf
    x = tf.convert_to_tensor(preprocess(img, meta))
    layer = model.get_layer(meta["gradcam_layer"])
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.outputs[0]])
    with tf.GradientTape() as tape:
        conv, prob = grad_model(x, training=False)
        conv = tf.cast(conv, tf.float32)
        prob = tf.clip_by_value(tf.cast(prob, tf.float32), 1e-7, 1 - 1e-7)
        logit = tf.math.log(prob / (1 - prob))
    grads = tape.gradient(logit, conv)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    cam = cv2.resize(cam, (int(meta["img_size"]),) * 2, interpolation=cv2.INTER_LINEAR)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return float(prob.numpy().ravel()[0]), cam


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> Image.Image:
    s = cam.shape[0]
    base = np.asarray(img.convert("RGB").resize((s, s)), dtype=np.float32)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32)
    return Image.fromarray(np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8))


def sample_video_frames(data: bytes, n: int = 16) -> list[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        path = f.name
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1))).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    Path(path).unlink(missing_ok=True)
    return frames


# --------------------------------------------------------------------------- UI helpers
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def step(n: str, title: str, hint: str = ""):
    html(f'<div class="step"><span class="n">{n}</span><h3>{title}</h3><small>{hint}</small></div>')


def result_card(prob: float, thr: float, extra: str = ""):
    fake = prob >= thr
    cls = "fake" if fake else "real"
    verdict = "Terindikasi deepfake" if fake else "Tidak terindikasi deepfake"
    action = ("Disarankan verifikasi lanjutan: telusuri sumber asli dan bandingkan dengan rekaman lain."
              if fake else "Tidak ada sinyal kuat manipulasi wajah. Tetap periksa konteks dan sumber konten.")
    html(f"""
<div class="card result {cls}">
  <div class="label">Hasil analisis</div>
  <span class="chip {cls}"><span class="dot" style="background:currentColor"></span>{verdict}</span>
  <div class="score">{prob*100:.1f}<small>%</small></div>
  <div class="note">probabilitas deepfake{extra}</div>
  <div class="bar"><span style="width:{max(prob*100, 1):.1f}%"></span><i style="left:calc({thr*100:.1f}% - 1px)"></i></div>
  <div class="scale"><span>0</span><span>ambang {thr:.2f}</span><span>1</span></div>
  <div class="divider"></div>
  <div class="note"><b style="color:var(--ink)">Langkah berikutnya.</b> {action}</div>
  <div class="note" style="margin-top:8px;font-size:.8rem;color:var(--faint)">Penyaring awal, bukan bukti forensik. Keputusan akhir pada pemeriksa fakta.</div>
</div>""")


def no_model_card():
    html("""
<div class="empty">
  <div class="ic"></div>
  <h4>Model belum tersedia</h4>
  <div class="note">Tambahkan <code>models/model.keras</code> dan <code>models/model_meta.json</code>,<br>
  atau isi <code>MODEL_URL</code> &amp; <code>META_URL</code> di Secrets Streamlit Cloud.</div>
</div>""")


def image_pair(face: Image.Image, cam_img: Image.Image | None, found: bool):
    with st.container(key="card_pair"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            html('<div class="label">Wajah terdeteksi</div>')
            st.image(face, width="stretch")
        with c2:
            html('<div class="label">Peta Grad-CAM</div>')
            st.image(cam_img if cam_img is not None else face.convert("L"), width="stretch")
        html('<div class="legend"></div><div class="scale"><span>pengaruh rendah</span><span>pengaruh tinggi</span></div>')
        if not found:
            html('<div class="note" style="margin-top:10px">Wajah tidak terdeteksi otomatis; seluruh citra digunakan.</div>')


# --------------------------------------------------------------------------- halaman
model, meta = load_model()
thr = float(meta.get("threshold", 0.5))
ready = model is not None
MODEL_LABEL = {"xception": "Xception", "efficientnetb0": "EfficientNet-B0", "mesonet": "MesoNet"}.get(meta["model_name"], meta["model_name"])

html(f"""
<div class="nav">
  <div class="brand"><div class="mark"></div>{APP_NAME}</div>
  <span class="pill">Prototipe riset · UMN</span>
</div>
<div class="hero">
  <div class="eyebrow">Deepfake face inspector</div>
  <h1>Periksa wajah sebelum<br>konten menyebar.</h1>
  <p>Unggah citra wajah atau video pendek. {APP_NAME} memberi skor kemungkinan deepfake dan menandai area wajah
  yang paling memengaruhi keputusan model.</p>
  <div class="stats">
    <div class="stat"><b>Status</b><span><span class="dot" style="background:{'#3DDC97' if ready else '#F5B544'}"></span>{'Model siap' if ready else 'Menunggu model'}</span></div>
    <div class="stat"><b>Model</b><span>{MODEL_LABEL}</span></div>
    <div class="stat"><b>Ambang</b><span>{thr:.2f}</span></div>
    <div class="stat"><b>Penjelasan</b><span>Grad-CAM</span></div>
  </div>
</div>""")

tab_img, tab_vid, tab_about = st.tabs(["Citra", "Video", "Tentang"])

with tab_img:
    step("01", "Unggah citra", "JPG, PNG, atau WEBP · wajah dipotong otomatis")
    up = st.file_uploader("Unggah citra wajah", type=["jpg", "jpeg", "png", "webp"], key="img", label_visibility="collapsed")
    if up is not None:
        img = Image.open(io.BytesIO(up.read()))
        face, found = crop_face(img)
        step("02", "Hasil analisis")
        left, right = st.columns([1, 1.25], gap="large")
        if not ready:
            with left: no_model_card()
            with right: image_pair(face, None, found)
        else:
            with st.spinner("Menganalisis…"):
                prob, cam = predict_with_gradcam(model, meta, face)
            with left: result_card(prob, thr)
            with right: image_pair(face, overlay(face, cam), found)

with tab_vid:
    step("01", "Unggah video", "MP4, MOV, atau AVI · maks. 50 MB")
    upv = st.file_uploader("Unggah video pendek", type=["mp4", "mov", "avi"], key="vid", label_visibility="collapsed")
    n_frames = st.slider("Jumlah frame yang dianalisis", 8, 32, 16, 4)
    if upv is not None:
        frames = sample_video_frames(upv.read(), n_frames)
        step("02", "Hasil analisis", f"{len(frames)} frame diambil merata")
        if not frames:
            st.error("Video tidak dapat dibaca.")
        elif not ready:
            no_model_card()
        else:
            probs, faces, cams = [], [], []
            bar = st.progress(0.0, text="Menganalisis frame…")
            for i, fr in enumerate(frames):
                f, _ = crop_face(fr)
                p, c = predict_with_gradcam(model, meta, f)
                probs.append(p); faces.append(f); cams.append(c)
                bar.progress((i + 1) / len(frames), text=f"Frame {i+1}/{len(frames)}")
            bar.empty()
            left, right = st.columns([1, 1.25], gap="large")
            with left:
                result_card(float(np.mean(probs)), thr, extra=f" · rata-rata {len(probs)} frame")
            with right:
                with st.container(key="card_chart"):
                    html('<div class="label">Skor per frame</div>')
                    st.line_chart({"skor": probs}, height=250, color="#13285A")
            step("03", "Frame paling mencurigakan", "diurutkan dari skor tertinggi")
            with st.container(key="card_top"):
                for col, i in zip(st.columns(4), np.argsort(probs)[::-1][:4]):
                    col.image(overlay(faces[i], cams[i]), caption=f"frame {i+1} · {probs[i]*100:.0f}%", width="stretch")

with tab_about:
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        html(f"""
<div class="card">
  <div class="label">Tentang</div>
  <h3 style="margin:0 0 10px;font-size:1.25rem;letter-spacing:-.01em">Alat bantu, bukan hakim.</h3>
  <p class="note">{APP_NAME} adalah prototipe skripsi Sistem Informasi, Universitas Multimedia Nusantara, yang
  membandingkan Xception, EfficientNet-B0, dan MesoNet dengan Grad-CAM untuk deteksi deepfake yang tahan terhadap
  kompresi media sosial. Model dilatih pada FaceForensics++ (C23).</p>
  <div class="divider"></div>
  <div class="label">Keterbatasan</div>
  <p class="note">Klasifikasi tingkat frame; audio tidak dinilai. Kinerja dapat turun pada metode manipulasi baru
  atau kompresi berat. Berkas yang diunggah tidak disimpan.</p>
</div>""")
    with b:
        html(f"""
<div class="card">
  <div class="label">Konfigurasi aktif</div>
  <p class="note" style="font-family:'JetBrains Mono',monospace;line-height:2">
  model&nbsp;&nbsp;&nbsp;&nbsp;{MODEL_LABEL}<br>ambang&nbsp;&nbsp;&nbsp;{thr:.2f}<br>layer&nbsp;&nbsp;&nbsp;&nbsp;{meta['gradcam_layer']}<br>input&nbsp;&nbsp;&nbsp;&nbsp;{meta['img_size']}×{meta['img_size']}</p>
</div>""")

html(f'<div class="foot"><span>{APP_NAME} · prototipe riset</span><span>Bukan alat pembuktian forensik</span></div>')
