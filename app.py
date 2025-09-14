import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import speech_recognition as sr

# ------------------------------
# Load Text Emotion Model
# ------------------------------
text_model_name = "j-hartmann/emotion-english-distilroberta-base"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
text_emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def predict_text_emotion(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = text_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    emotion = text_emotion_labels[pred_idx]
    confidence = probs[0][pred_idx].item()
    return emotion, confidence

# ------------------------------
# Load Emoji Emotion Model
# ------------------------------
emoji_model_name = "cardiffnlp/twitter-roberta-base-emoji"
emoji_tokenizer = AutoTokenizer.from_pretrained(emoji_model_name)
emoji_model = AutoModelForSequenceClassification.from_pretrained(emoji_model_name)

emoji_labels = [
    "😂", "😍", "😭", "😊", "😒", "💕", "👌", "😘", "😁", "😩",
    "🔥", "🙏", "😏", "😉", "🙌", "😔", "💪", "😷", "👏", "😃"
]
emoji_to_emotion = {
    "😂": "joy", "😍": "love", "😭": "sadness", "😊": "happiness", "😒": "disapproval",
    "💕": "affection", "👌": "approval", "😘": "affection", "😁": "cheerful", "😩": "tired",
    "🔥": "excitement", "🙏": "gratitude", "😏": "smug", "😉": "playful", "🙌": "celebration",
    "😔": "disappointment", "💪": "strength", "😷": "sick", "👏": "praise", "😃": "happy"
}

def predict_emoji_emotion(emoji_text):
    inputs = emoji_tokenizer(emoji_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = emoji_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs).item()
        emoji = emoji_labels[pred_class]
        label = emoji_to_emotion.get(emoji, "unknown")
        confidence = probs[0][pred_class].item()
        return emoji, label, confidence

# ------------------------------
# Voice Input
# ------------------------------
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Audio captured, transcribing...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("😓 Could not understand the audio.")
        except sr.RequestError:
            st.error("⚠️ Could not request results. Check internet.")
        except sr.WaitTimeoutError:
            st.error("⌛ Timeout: No speech detected.")
    return None

# ------------------------------
# Streamlit Dashboard UI
# ------------------------------
st.set_page_config(page_title="🧠 Multi-Input Emotion Detector", layout="wide", page_icon="🧠")
st.title("🧠 Multi-Input Emotion Detection Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Input Mode:", ["Text", "Voice", "Emoji"])

# Emotion color mapping
emotion_colors = {
    "anger": "red", "disgust": "green", "fear": "purple", "joy": "yellow",
    "neutral": "gray", "sadness": "blue", "surprise": "orange",
    "love": "pink", "happiness": "gold", "disapproval": "brown",
    "affection": "violet", "approval": "lime", "cheerful": "lightgreen",
    "tired": "lightblue", "excitement": "orangered", "gratitude": "turquoise",
    "smug": "magenta", "playful": "cyan", "celebration": "goldenrod",
    "disappointment": "darkblue", "strength": "darkgreen", "sick": "gray",
    "praise": "lightyellow", "happy": "gold"
}

def show_emotion_card(emotion, confidence):
    color = emotion_colors.get(emotion.lower(), "lightgray")
    st.markdown(f"""
        <div style='padding: 20px; border-radius: 15px; background-color: {color}; text-align:center;'>
            <h2 style='color:black'>{emotion.capitalize()}</h2>
            <h4 style='color:black'>Confidence: {confidence:.2%}</h4>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# Mode: Text Input
# ------------------------------
if mode == "Text":
    st.subheader("✏️ Text Input")
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze Text Emotion"):
        if user_input.strip() != "":
            with st.spinner("Analyzing..."):
                emotion, confidence = predict_text_emotion(user_input)
            show_emotion_card(emotion, confidence)
        else:
            st.warning("⚠️ Please enter some text.")

# ------------------------------
# Mode: Voice Input
# ------------------------------
elif mode == "Voice":
    st.subheader("🎤 Voice Input")
    if st.button("Start Recording"):
        voice_text = get_voice_input()
        if voice_text:
            st.markdown(f"**You said:** `{voice_text}`")
            with st.spinner("Analyzing..."):
                emotion, confidence = predict_text_emotion(voice_text)
            show_emotion_card(emotion, confidence)

# ------------------------------
# Mode: Emoji Input
# ------------------------------
elif mode == "Emoji":
    st.subheader("😊 Emoji Input")
    emoji_input = st.text_input("Enter an Emoji:")
    if st.button("Analyze Emoji"):
        if emoji_input.strip() != "":
            try:
                with st.spinner("Analyzing..."):
                    predicted_emoji, emotion, confidence = predict_emoji_emotion(emoji_input)
                st.markdown(f"**Closest Matching Emoji:** {predicted_emoji}")
                show_emotion_card(emotion, confidence)
            except Exception:
                st.error("❌ Could not process that emoji.")
        else:
            st.warning("⚠️ Please enter an emoji.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Developed with ❤️ using **Streamlit & Hugging Face Transformers**")
