import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_model.keras")

model = load_model()

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

st.title("🧠 Brain Tumor MRI Classification")
st.markdown("---")

st.write("Upload an MRI scan and the model will predict the tumor type.")

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    prediction = model.predict(img, verbose=0)

    pred = np.argmax(prediction)

    confidence = np.max(prediction)

    with col2:
        st.success(f"Prediction\n\n### {class_names[pred]}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

    st.markdown("---")

    st.subheader("Prediction Probability")

    for i in range(4):
        st.progress(float(prediction[0][i]))
        st.write(
            f"**{class_names[i]}** : {prediction[0][i]*100:.2f}%"
        )