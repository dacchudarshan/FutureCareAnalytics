# FutureCareAnalytics
AI-powered Streamlit healthcare dashboard with ML-based risk prediction, multi-language support, and voice-enabled insights.

# 🩺 Future Care Analytics – Predictive Healthcare Dashboard

A comprehensive multi-language healthcare analytics application that provides personalized health insights through predictive modeling and interactive visualizations.

## ✨ Features

- **Multi-Language Support**: English, Hindi, and Kannada language support for accessibility
- **Health Parameter Analysis**: Interactive sliders for age, blood pressure, glucose level, cholesterol, and heart rate
- **Symptom Analysis**: Natural language symptom input with intelligent matching and remedies
- **Risk Assessment**: Machine learning-based prediction model for health risk evaluation
- **Interactive Visualizations**: Real-time charts and graphs for health metrics
- **Text-to-Speech**: Audio feedback in multiple languages for accessibility
- **Personalized Insights**: Custom health recommendations based on input parameters

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn (Random Forest Classifier)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Audio**: pyttsx3, gTTS (Google Text-to-Speech)
- **Language Detection**: langdetect

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- pyttsx3
- langdetect
- gTTS

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FutureCareAnalytics.git
cd FutureCareAnalytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### How to Use:
1. Select your preferred language
2. Adjust health parameter sliders (age, blood pressure, glucose, cholesterol, heart rate)
3. Describe your symptoms in the text area
4. View personalized health insights and recommendations
5. Listen to audio feedback in your selected language

## 🏥 Health Parameters

- **Age**: 20-80 years
- **Blood Pressure**: 80-200 mmHg
- **Glucose Level**: 70-300 mg/dL
- **Cholesterol**: 150-350 mg/dL
- **Heart Rate**: 40-150 bpm

## 🗣️ Supported Languages

- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Kannada

## ⚕️ Disclaimer

**This application is for informational and educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 👨‍💻 Author

Darshit Patel

## 📞 Support

For support, please open an issue on the GitHub repository or contact the maintainer.

---

**Note**: This dashboard uses machine learning models for health risk assessment. Results are predictions and should not replace professional medical consultation.
