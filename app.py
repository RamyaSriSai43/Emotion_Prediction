import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import speech_recognition as sr

# Load text emotion model
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


# Load emoji emotion model
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


# Voice input handling
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
            st.error("😓 Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("⚠️ Could not request results. Check internet.")
        except sr.WaitTimeoutError:
            st.error("⌛ Timeout: No speech detected.")
    return None


# Streamlit UI
st.set_page_config(page_title="Multi-Input Emotion Detector", layout="centered")

st.markdown("## 🧠 Emotion Prediction App")
mode = st.radio("Select Input Mode:", ["Text", "Voice", "Emoji"])

if mode == "Text":
    user_input = st.text_input("✏️ Enter your text:")
    if user_input:
        with st.spinner("Analyzing..."):
            emotion, confidence = predict_text_emotion(user_input)
        st.markdown(f"**Predicted Emotion:** {emotion.capitalize()}")
        st.markdown(f"**Confidence:** {confidence:.2%}")

elif mode == "Voice":
    if st.button("🎤 Start Recording"):
        voice_text = get_voice_input()
        if voice_text:
            st.markdown(f"**You said:** `{voice_text}`")
            with st.spinner("Analyzing..."):
                emotion, confidence = predict_text_emotion(voice_text)
            st.markdown(f"**Predicted Emotion:** {emotion.capitalize()}")
            st.markdown(f"**Confidence:** {confidence:.2%}")

elif mode == "Emoji":
    emoji_input = st.text_input("😊 Enter an Emoji:")
    if emoji_input:
        try:
            with st.spinner("Analyzing..."):
                predicted_emoji, emotion, confidence = predict_emoji_emotion(emoji_input)
            st.markdown(f"**Predicted Emotion:** {emotion.capitalize()}")
            st.markdown(f"**Closest Matching Emoji:** {predicted_emoji}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
        except Exception as e:
            st.error("❌ Could not process that emoji.")
