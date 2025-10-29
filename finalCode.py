import os
import random
import pickle
import gradio as gr
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import whisper
from tensorflow.keras.models import load_model
import re
import shutil

# ==============================
# Add ffmpeg path (if needed)
# ==============================
os.environ["PATH"] += os.pathsep + r"C:\Users\HP\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

# ==============================
# Config fixed audio folder
# ==============================
SAVE_FOLDER = r"C:\Users\HP\Documents\ArabicPracticeAudio"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==============================
# Load models / data
# ==============================
model_cnn = load_model("mfcc_cnn_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

letters_to_practice = [
    "ا","ب","ت","ث","ج","ح","خ","د","ذ",
    "ر","ز","س","ش","ص","ض","ط","ظ",
    "ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي"
]
current_index = 0

# Load words
try:
    df_words = pd.read_csv(r"C:\Users\HP\Downloads\archive (10)\arabic_dialects_stopwords.csv")
    word_list = df_words["word"].dropna().astype(str).tolist()
    if not word_list:
        word_list = ["مرحبا", "أهلا", "كيف", "حالك", "نعم", "لا"]
except Exception:
    word_list = ["مرحبا", "أهلا", "كيف", "حالك", "نعم", "لا"]

# Load Whisper model (medium for better accuracy)
model_whisper = whisper.load_model("medium")

# ==============================
# Utilities
# ==============================
def denoise_audio(y, sr, use_denoise=True):
    if not use_denoise:
        return y
    try:
        import noisereduce as nr
        y_trim, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trim) < int(0.2 * sr):
            y_trim = y
        y_nr = nr.reduce_noise(y=y_trim, sr=sr, stationary=False, prop_decrease=0.9)
        return y_nr
    except Exception:
        return y

def normalize_arabic(text):
    text = re.sub(r"[^\u0621-\u064A\s]", "", text)
    return text.strip()

def save_audio_fixed_path(audio_path):
    """Save user audio to a fixed path and normalize length."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    fixed_path = os.path.join(SAVE_FOLDER, "recorded.wav")
    y, sr = librosa.load(audio_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=25)
    sf.write(fixed_path, y, sr)
    return fixed_path

# ==============================
# CNN preprocessing
# ==============================
def preprocess_for_cnn(audio_path, remove_noise=True):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Remove silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Denoise if requested
    if remove_noise:
        y = denoise_audio(y, sr, use_denoise=True)

    # Normalize amplitude to [-1,1]
    max_val = np.max(np.abs(y)) + 1e-9
    y = y / max_val

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Pad or truncate to max_len
    max_len = 50
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    # Expand dims for CNN input
    mfcc = np.expand_dims(mfcc.astype("float32"), axis=(0,-1))
    return mfcc

# ==============================
# CNN Letter Practice
# ==============================
def practice_letter(audio_path, remove_noise):
    global current_index
    if not audio_path or not os.path.exists(audio_path):
        return f"قل الحرف: {letters_to_practice[current_index]}", "⚠️ يرجى تسجيل صوتك أولاً."

    try:
        expected_letter = letters_to_practice[current_index]
        mfcc = preprocess_for_cnn(audio_path, remove_noise)
        pred_probs = model_cnn.predict(mfcc, verbose=0)
        pred_class = np.argmax(pred_probs)
        pred_letter = label_encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred_probs))

        # Adjust threshold: only accept predictions above 50% confidence
        if confidence < 0.5:
            message = f"⚠️ لم يتم التعرف بوضوح. حاول مرة أخرى!"
        elif pred_letter == expected_letter:
            message = f"✅ صح! '{pred_letter}' الثقة: {confidence*100:.2f}%"
            current_index = (current_index + 1) % len(letters_to_practice)
        else:
            message = f"❌ خطأ! قلت '{pred_letter}'. حاول مرة أخرى!"

        next_prompt = f"قل الحرف التالي: {letters_to_practice[current_index]}"
        return next_prompt, message

    except Exception as e:
        return f"قل الحرف: {letters_to_practice[current_index]}", f"⚠️ خطأ: {str(e)}"

# ==============================
# Word Practice (Whisper)
# ==============================
def get_random_word():
    return random.choice(word_list) if word_list else "مرحبا"

def check_word_pronunciation(audio_path, expected_text):
    fixed_path = save_audio_fixed_path(audio_path)
    if not fixed_path:
        return "⚠️ يرجى تسجيل صوتك أولاً."

    try:
        # Whisper transcription with beam search for better accuracy
        result = model_whisper.transcribe(fixed_path, language="ar", beam_size=5, temperature=0.0)
        recognized_text = normalize_arabic(result.get("text") or "")
    except Exception as e:
        return f"Erreur Whisper: {str(e)}"

    expected_text_norm = normalize_arabic(expected_text)
    if not recognized_text:
        return "❌ لم يتم التعرف على الصوت."
    if recognized_text == expected_text_norm:
        return f"✅ صحيح! لقد قلت: {recognized_text}"
    else:
        return f"❌ خطأ! كان يجب أن تقول: {expected_text}، لكن قلت: {recognized_text}"

# ==============================
# Gradio Interface
# ==============================
custom_css = """
.gradio-container { padding: 32px 16px; background: linear-gradient(0deg, rgba(0,0,0,0.25), rgba(0,0,0,0.25)),
                url('https://talkao.com/wp-content/uploads/2024/04/Arquitectura-islamica-Abu-Dhabi-maravillas-arquitectonicas-del-mundo-arabe.webp') no-repeat center center;
                background-size: cover; }
.app-card { direction: rtl; width: min(1000px, 95vw); margin: 18px auto; padding:28px; border-radius:24px;
            background: rgba(255,255,255,0.78); box-shadow:0 12px 30px rgba(0,0,0,0.28); backdrop-filter:blur(8px);
            border:1px solid rgba(255,255,255,0.45); }
.title { text-align:center; margin:0 0 14px 0; }
.title h1 { margin:0; font-size:clamp(28px,4vw,40px); font-weight:800; letter-spacing:.3px;
           background:linear-gradient(90deg,#5b21b6,#0ea5e9); -webkit-background-clip:text;
           background-clip:text; color:transparent !important; text-shadow:0 2px 10px rgba(0,0,0,0.15); }
.app-card label { color:#4b0082 !important; font-weight:700 !important; font-size:16px; }
.pretty-input textarea { background: linear-gradient(135deg,#f9f0ff,#f0f8ff); border:2px solid #6a5acd !important;
                        border-radius:16px !important; padding:14px !important; font-size:18px !important; color:#222 !important;
                        box-shadow: inset 0 3px 8px rgba(0,0,0,0.07); }
.pretty-audio { border:2px solid #6a5acd !important; border-radius:18px !important; padding:10px !important;
                background:#ffffffd9 !important; box-shadow:0 6px 18px rgba(0,0,0,0.12); }
button { border-radius:12px !important; font-weight:700 !important; transition: transform .15s ease, box-shadow .15s ease; }
button:hover { transform:translateY(-1px); box-shadow:0 8px 18px rgba(106,90,205,0.35); }
.app-card > *:not(:last-child) { margin-bottom:14px; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<div class='title'><h1>🎤 التدريب العربي: الحروف والكلمات 🎶</h1></div>")

    # Letters tab
    with gr.Tab("📖 تدريب الحروف بالصوت"):
        with gr.Group(elem_classes=["app-card"]):
            prompt = gr.Textbox(label="التعليمات", value=f"قل الحرف: {letters_to_practice[current_index]}", interactive=False, elem_classes=["pretty-input"])
            feedback = gr.Textbox(label="النتيجة", interactive=False, elem_classes=["pretty-input"])
            denoise_chk = gr.Checkbox(label="🧼 إزالة ضوضاء الخلفية", value=True)
            audio_input_mic = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ تسجيل من الميكروفون", elem_classes=["pretty-audio"])
            audio_input_file = gr.Audio(sources=["upload"], type="filepath", label="📂 تحميل ملف صوتي", elem_classes=["pretty-audio"])

        audio_input_mic.change(practice_letter, inputs=[audio_input_mic, denoise_chk], outputs=[prompt, feedback])
        audio_input_file.change(practice_letter, inputs=[audio_input_file, denoise_chk], outputs=[prompt, feedback])

    # Words tab
    with gr.Tab("📝 تدريب الكلمات/العبارات"):
        with gr.Group(elem_classes=["app-card"]):
            word_display = gr.Textbox(label="كلمة/عبارة لتدريب النطق", interactive=False, value=get_random_word())
            audio_input_word = gr.Audio(sources=["microphone","upload"], type="filepath", label="🎙️ سجل صوتك أو حمل ملفًا", elem_classes=["pretty-audio"])
            feedback_word = gr.Textbox(label="النتيجة", interactive=False, elem_classes=["pretty-input"])
            btn_check = gr.Button("تحقق من النطق")
            btn_new = gr.Button("كلمة جديدة")

        btn_check.click(check_word_pronunciation, inputs=[audio_input_word, word_display], outputs=feedback_word)
        btn_new.click(lambda: get_random_word(), outputs=word_display)

demo.launch()
