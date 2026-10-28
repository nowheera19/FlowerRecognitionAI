
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="BloomAI - Flower Recognition",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 BloomAI")
st.subheader("Flower Species Recognition Using Deep Learning")

st.write(
    "Upload a flower image and let our AI model "
    "predict its species."
)

# -------------------------------
# 2. File Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "flower_model.keras"
CLASSES_PATH = BASE_DIR / "class_names.json"


# -------------------------------
# 3. Verify Required Files
# -------------------------------
def verify_file(path):
    if not path.is_file():
        st.error(f"File not found: {path.name}")
        st.stop()

    if path.stat().st_size == 0:
        st.error(f"File is empty: {path.name}")
        st.stop()


verify_file(MODEL_PATH)
verify_file(CLASSES_PATH)


# -------------------------------
# 4. Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    model = load_model()
    class_names = load_classes()

except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# -------------------------------
# 5. Validate Class Names
# -------------------------------
if not isinstance(class_names, list):
    st.error("class_names.json must contain a list of class names.")
    st.stop()

if len(class_names) != model.output_shape[-1]:
    st.error(
        "Class names do not match the model output. "
        "Please check class_names.json."
    )
    st.stop()


# -------------------------------
# 6. Upload Flower Image
# -------------------------------
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose a flower photograph",
    type=["jpg", "jpeg", "png"]
)


# -------------------------------
# 7. Predict Flower Species
# -------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Uploaded flower",
                use_container_width=True
            )

        with st.spinner("Recognizing your flower..."):

            # Resize the image to the model's input size
            resized = image.resize((224, 224))

            # Convert image into a NumPy array
            array = np.asarray(
                resized,
                dtype=np.float32
            )

            # Add batch dimension
            array = np.expand_dims(array, axis=0)

            # IMPORTANT:
            # Preprocessing is already included inside
            # the trained model. Do not preprocess again here.
            probabilities = model.predict(
                array,
                verbose=0
            )[0]

        # Get the three highest predictions
        top_indices = np.argsort(probabilities)[::-1][:3]

        best_index = int(top_indices[0])
        confidence = float(probabilities[best_index])

        # -------------------------------
        # 8. Display Main Prediction
        # -------------------------------
        with right:
            st.markdown("### 🌼 Prediction")

            st.success(
                f"Predicted flower: "
                f"{class_names[best_index].title()}"
            )

            st.metric(
                "Prediction confidence",
                f"{confidence:.1%}"
            )

            st.caption(
                "Confidence is the model's estimate, "
                "not a guarantee that the prediction is correct."
            )

        # -------------------------------
        # 9. Display Top 3 Predictions
        # -------------------------------
        st.markdown("---")
        st.markdown("### 📊 Top 3 Predictions")

        chart_data = {
            class_names[int(i)].title():
                float(probabilities[i])
            for i in top_indices
        }

        st.bar_chart(chart_data)

        for i in top_indices:
            st.write(
                f"**{class_names[int(i)].title()}**: "
                f"{probabilities[i]:.2%}"
            )

    except Exception as e:
        st.error(f"Unable to process image: {e}")

else:
    st.info("Upload a flower photo to get started.")


# -------------------------------
# 10. Footer
# -------------------------------
st.markdown("---")

st.caption(
    "BloomAI | Built with Python, TensorFlow and Streamlit"
)
