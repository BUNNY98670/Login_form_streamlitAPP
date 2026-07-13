import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.title("📰 Fake News Detector")

news = st.text_area("Enter News Text")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text")
    else:
        # Transform input
        data = vectorizer.transform([news])

        # Predict
        prediction = model.predict(data)

        # Output
        if prediction[0] == 1:
            st.success("✅ TRUE News")
        else:
            st.error("❌ FAKE News")