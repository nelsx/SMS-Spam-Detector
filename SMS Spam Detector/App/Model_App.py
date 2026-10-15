import joblib
import streamlit as st
from pathlib import Path

CURRENT_DIR = Path.cwd()

model = joblib.load(CURRENT_DIR.parent / "Artifacts" / "Logistic_Regression_SMS_Spam.joblib")
vectorizer = joblib.load(CURRENT_DIR.parent / "Artifacts" / "TfidfVectorizer_SMS_Spam.joblib")

#page Configuration
st.set_page_config(page_title="SMS Spam Detector",
                   page_icon="📩",
                   layout="centered")
#Title
st.title("SMS Spam Detector")

st.write(
    "Enter SMS message below. This model predicts whether the message is spam or not. \nPlease note that this model is trained on a specific dataset and may not generalize well to all types of messages."
)

#SMS input
message = st.text_area("Enter your SMS message here:",
                       height=150,
                       placeholder="Example: Congratulations! You've won a free ticket to Bahamas. Call now!")

#prediction button
if st.button("Predict", use_container_width=True):
    if message.strip() == "":
        st.warning("Please enter an SMS message to predict.")
    else:
        # Preprocess the input message
        message_vectorized = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_vectorized)[0]

        # Get prediction probability
        probability = model.predict_proba(message_vectorized)[0]

        # Display result
        if prediction == 1:
            confidence = probability[1] * 100

            st.error("🚨 SPAM MESSAGE DETECTED! 🚨")

            st.write(f"Prediction Confidence: {confidence:.2f}%")
        else:
            st.success("✅ NOT SPAM.")

            st.write(f"Prediction Confidence: {probability[0] * 100:.2f}%")