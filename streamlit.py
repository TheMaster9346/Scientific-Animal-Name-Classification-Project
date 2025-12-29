import streamlit as st
from fastai.vision.all import *

st.title("Scientific Animal Name Prediction")
st.text("Built by Rafa Kim")


def extract_animal_name(file_path):
    file_parts = str(file_path).split("/")
    folder_name = file_parts[-2]

    return folder_name


scientific_animal_name_model = load_learner("scientific_animal_name_model.pkl")

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = scientific_animal_name_model.predict(img)
    return pred_class


uploaded_file = st.file_uploader("Upload an image of a animal....", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Digit: {prediction}")


st.text("Built with Streamlit and FastAI")