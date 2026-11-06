import streamlit as st
import joblib

# Model aur vectorizer load karein
model = joblib.load('random_forest_text_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("SMS Spam Classifier using Random Forest")
st.write("Apna message enter karein taake model predict kar sakay ke yeh Spam hai ya Normal!")

# User input text box
user_input = st.text_area("Yahan message likhein:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Text ko transform karein
        input_vector = vectorizer.transform([user_input])
        
        # Prediction karein
        prediction = model.predict(input_vector)
        
        # Result show karein (0 = Normal, 1 = Spam)
        if prediction[0] == 1:
            st.error("🚨 Yeh message **SPAM** hai!")
        else:
            st.success("✅ Yeh message **Normal (Ham)** hai!")
    else:
        st.warning("Pehle kuch text toh enter karein!")
