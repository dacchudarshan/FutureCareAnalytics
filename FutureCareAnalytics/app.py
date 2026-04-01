import streamlit as st
import pandas as pd
import numpy as np
import pyttsx3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
from langdetect import detect
from gtts import gTTS
import tempfile
import matplotlib.pyplot as plt
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🏥 Future Care Analytics", page_icon="🩺", layout="wide")
st.title("🩺 Future Care Analytics – Predictive Healthcare Dashboard")
st.markdown("Type your symptoms and enter your health data to get personalized health insights with interactive visualizations!")

# -----------------------------
# LANGUAGE SELECTION
# -----------------------------
languages = ["English", "Hindi", "Kannada"]
lang_map = {"English": "en", "Hindi": "hi", "Kannada": "kn"}
preferred_lang = st.selectbox("Select the language :", languages)

# -----------------------------
# HEALTH DATA INPUT (SLIDERS)
# -----------------------------
st.subheader("🧍 Enter Your Health Parameters")
age = st.slider("Age", 20, 80, 40)
bp = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
glucose = st.slider("Glucose Level (mg/dL)", 70, 300, 110)
chol = st.slider("Cholesterol (mg/dL)", 150, 350, 220)
hr = st.slider("Heart Rate (bpm)", 40, 150, 80)

# -----------------------------
# TYPED SYMPTOM INPUT
# -----------------------------
st.subheader("📝 Describe Your Symptoms")
symptom_text = st.text_area("Type your symptoms here:")

# -----------------------------
# MULTI-LANGUAGE SYMPTOM TO SOLUTION
# -----------------------------
multi_symptom_map = {
    "en": {"headache": "Rest, compress, drink water",
           "fever": "Drink fluids, rest, take paracetamol",
           "tired": "Rest and eat nutritious food",
           "pain": "Use hot water, rest the area"},
    "hi": {"सिरदर्द": "आराम करें, गर्म/ठंडी सिकाई, पानी पिएँ",
           "बुखार": "पानी पिएँ, आराम करें, पैरासिटामोल लें",
           "थकान": "आराम करें और पौष्टिक भोजन लें",
           "दर्द": "गर्म पानी लगाएँ और आराम करें"},
    "kn": {"ತಲೆನೋವು": "ಉರಿಯುವ ಅಥವಾ ತಂಪು ಸುಣ್ಣ, ವಿಶ್ರಾಂತಿ, ನೀರು ಕುಡಿಯಿರಿ",
           "ಜ್ವರ": "ಹೆಚ್ಚು ನೀರು ಕುಡಿಯಿರಿ, ವಿಶ್ರಾಂತಿ, ಪ್ಯಾರಾಸಿಟಾಮೋಲ್ ತೆಗೆದುಕೊಳ್ಳಿ",
           "ತಲುಪಿದ": "ವಿಶ್ರಾಂತಿ, ಹಣ್ಣು ತರಕಾರಿ ತಿನ್ನಿ",
           "ನೋವು": "ತಾಪಮಾನ ನೀರು ಬಳಸಿ, ವಿಶ್ರಾಂತಿ"}
}

# -----------------------------
# SYSTEM MESSAGES
# -----------------------------
system_messages = {
    "en": {
        "greeting": "Hello! Let's review your health report.",
        "high_risk": "⚠️ High Risk Detected!",
        "low_risk": "✅ Low Risk",
        "blood_high": "🩺 Blood pressure high. Reduce salt, avoid stress.",
        "glucose_high": "🩺 Sugar level high. Avoid sweets.",
        "chol_high": "🩺 Cholesterol high. Eat vegetables, avoid oily food.",
        "hr_high": "🩺 Heart rate high. Rest and avoid caffeine.",
        "hr_low": "🩺 Heart rate low. Eat healthy and monitor.",
        "age_risk": "🩺 Age-related risk detected. Maintain healthy lifestyle.",
        "reported_symptoms": "📝 Reported symptoms:",
        "suggested_remedies": "💡 Suggested Remedies:",
        "normal": "🎉 Your health parameters are within normal range."
    },
    "hi": {
        "greeting": "नमस्ते! आइए आपकी स्वास्थ्य रिपोर्ट देखें।",
        "high_risk": "⚠️ उच्च जोखिम पाया गया!",
        "low_risk": "✅ जोखिम कम",
        "blood_high": "🩺 उच्च रक्तचाप। नमक कम करें, तनाव से बचें।",
        "glucose_high": "🩺 शुगर लेवल उच्च। मिठाई से बचें।",
        "chol_high": "🩺 कोलेस्ट्रॉल उच्च। सब्जियाँ खाएँ, तैलीय भोजन से बचें।",
        "hr_high": "🩺 हृदय गति उच्च। आराम करें और कैफीन से बचें।",
        "hr_low": "🩺 हृदय गति कम। स्वस्थ आहार लें और मॉनिटर करें।",
        "age_risk": "🩺 आयु से संबंधित जोखिम। स्वस्थ जीवन शैली अपनाएँ।",
        "reported_symptoms": "📝 रिपोर्ट किए गए लक्षण:",
        "suggested_remedies": "💡 सुझाए गए उपचार:",
        "normal": "🎉 आपकी स्वास्थ्य स्थिति सामान्य है।"
    },
    "kn": {
        "greeting": "ನಮಸ್ಕಾರ! ನಿಮ್ಮ ಆರೋಗ್ಯ ವರದಿಯನ್ನು ಪರಿಶೀಲಿಸೋಣ.",
        "high_risk": "⚠️ ಹೆಚ್ಚಿನ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ!",
        "low_risk": "✅ ಕಡಿಮೆ ಅಪಾಯ",
        "blood_high": "🩺 ರಕ್ತದೊತ್ತಡ ಹೆಚ್ಚಾಗಿದೆ. ಉಪ್ಪು ಕಡಿಮೆ ಮಾಡಿ, ಒತ್ತಡ ತಪ್ಪಿಸಿ.",
        "glucose_high": "🩺 ರಕ್ತದಲ್ಲಿನ ಸಕ್ಕರೆ ಹೆಚ್ಚಾಗಿದೆ. ಸಿಹಿ ಆಹಾರ ತಪ್ಪಿಸಿ.",
        "chol_high": "🩺 ಕೊಲೆಸ್ಟ್ರಾಲ್ ಹೆಚ್ಚಾಗಿದೆ. ಹಣ್ಣು, ತರಕಾರಿ ತಿನ್ನಿ, ಎಣ್ಣೆयुक्त ಆಹಾರ ತಪ್ಪಿಸಿ.",
        "hr_high": "🩺 ಹೃದಯದ ತಾಳ ಹೆಚ್ಚಾಗಿದೆ. ವಿಶ್ರಾಂತಿ ಮಾಡಿ, ಕ್ಯಾಫಿನ್ ತಪ್ಪಿಸಿ.",
        "hr_low": "🩺 ಹೃದಯದ ತಾಳ ಕಡಿಮೆಯಾಗಿದೆ. ಆರೋಗ್ಯಕರ ಆಹಾರ ಸೇವಿಸಿ, ಗಮನಿಸಿ.",
        "age_risk": "🩺 ವಯಸ್ಸು ಸಂಬಂಧಿತ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ. ಆರೋಗ್ಯಕರ ಜೀವನ ಶೈಲಿ ಪಾಲಿಸಿ.",
        "reported_symptoms": "📝 ವರದಿಯಾದ ಲಕ್ಷಣಗಳು:",
        "suggested_remedies": "💡 ಸೂಚಿಸಿದ ಚಿಕಿತ್ಸೆ:",
        "normal": "🎉 ನಿಮ್ಮ ಆರೋಗ್ಯ ಸೂಚ್ಯಂಕಗಳು ಸಾಮಾನ್ಯ ಮಟ್ಟದಲ್ಲಿವೆ."
    }
}

# -----------------------------
# SIMULATED DATA & MODEL
# -----------------------------
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    "Age": np.random.randint(20, 80, n_samples),
    "BP": np.random.randint(80, 200, n_samples),
    "Glucose": np.random.randint(70, 300, n_samples),
    "HR": np.random.randint(40, 150, n_samples),
    "Chol": np.random.randint(150, 350, n_samples),
})
df["Risk"] = ((df["BP"] > 140) | (df["Glucose"] > 150) |
              (df["Chol"] > 250) | (df["Age"] > 65) |
              (df["HR"] > 110) | (df["HR"] < 55)).astype(int)

X = df[["Age", "BP", "Glucose", "HR", "Chol"]]
y = df["Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# STORY GENERATOR FUNCTION
# -----------------------------
def generate_story(age, bp, glucose, chol, hr, symptoms, preferred_lang):
    story = system_messages[preferred_lang]["greeting"] + "\n\n"

    # Predict risk
    risk = model.predict([[age, bp, glucose, hr, chol]])[0]
    prob = model.predict_proba([[age, bp, glucose, hr, chol]])[0][1]
    story += f"{system_messages[preferred_lang]['high_risk'] if risk==1 else system_messages[preferred_lang]['low_risk']} (Probability: {prob:.2f})\n"

    # Vitals warnings
    if bp > 140: story += system_messages[preferred_lang]["blood_high"] + "\n"
    if glucose > 150: story += system_messages[preferred_lang]["glucose_high"] + "\n"
    if chol > 250: story += system_messages[preferred_lang]["chol_high"] + "\n"
    if hr > 110: story += system_messages[preferred_lang]["hr_high"] + "\n"
    if hr < 55: story += system_messages[preferred_lang]["hr_low"] + "\n"
    if age > 65: story += system_messages[preferred_lang]["age_risk"] + "\n"

    # Symptom remedies
    if symptoms.strip():
        story += f"\n{system_messages[preferred_lang]['reported_symptoms']} {symptoms}\n"
        solutions_given = []
        for key, solution in multi_symptom_map[preferred_lang].items():
            matches = get_close_matches(symptoms.lower(), [key.lower()], cutoff=0.6)
            if matches:
                solutions_given.append(solution)
        if solutions_given:
            story += "\n" + system_messages[preferred_lang]["suggested_remedies"] + "\n"
            for sol in solutions_given:
                story += f"- {sol}\n"

    if not (bp>140 or glucose>150 or chol>250 or hr>110 or hr<55 or age>65):
        story += "\n" + system_messages[preferred_lang]["normal"] + "\n"

    return story

# -----------------------------
# SESSION STATE
# -----------------------------
if "records" not in st.session_state:
    st.session_state.records = []

# -----------------------------
# GENERATE STORY BUTTON
# -----------------------------
if st.button("Generate My Health Story"):
    story_text = generate_story(age, bp, glucose, chol, hr, symptom_text, lang_map[preferred_lang])
    st.subheader("📖 Your Personalized Health Story")
    st.write(story_text)

    # Audio in patient-selected language
    tts = gTTS(text=story_text, lang=lang_map[preferred_lang])
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    st.audio(tmp_file.name, format="audio/mp3")
    st.success(f"🎤 Your story is spoken in {preferred_lang}! You can listen again.")

    # Save record
    st.session_state.records.append({
        "Age": age,
        "BP": bp,
        "Glucose": glucose,
        "Cholesterol": chol,
        "HR": hr,
        "Symptoms": symptom_text,
        "Language": preferred_lang,
        "Story": story_text
    })

    # -----------------------------
    # VITALS VISUALIZATION
    # -----------------------------
    st.subheader("📊 Your Vitals Overview")
    vitals = {"Age": age, "BP": bp, "Glucose": glucose, "HR": hr, "Chol": chol}
    colors = []
    for k, v in vitals.items():
        if (k=="BP" and v>140) or (k=="Glucose" and v>150) or (k=="Chol" and v>250) or (k=="HR" and (v>110 or v<55)) or (k=="Age" and v>65):
            colors.append("red")
        else:
            colors.append("green")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(vitals.keys(), vitals.values(), color=colors)
    ax.set_ylabel("Value")
    ax.set_title("Patient Vitals vs Normal Range")
    st.pyplot(fig)

# -----------------------------
# VIEW SAVED PATIENT RECORDS
# -----------------------------
if st.session_state.records:
    st.subheader("🗂️ Saved Patient Records")
    st.dataframe(pd.DataFrame(st.session_state.records))