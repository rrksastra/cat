import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cat vs Dog - CNN",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Image Classification")
st.write("Convolutional Neural Network (CNN) + TensorFlow/Keras")


# =========================================================
# 2. MODEL PATH
# =========================================================

MODEL_PATH = Path(__file__).parent / "cat_dog_small.keras"


# =========================================================
# 3. CHECK MODEL
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        "❌ Model not found: cat_dog_small.keras"
    )

    st.info(
        "Please place cat_dog_small.keras "
        "in the same folder as app1.py"
    )

    st.stop()


# =========================================================
# 4. LOAD CNN MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# =========================================================
# 5. CLASSES
# =========================================================

class_names = [
    "cats",
    "dogs"
]


# =========================================================
# 6. MODEL INFORMATION
# =========================================================

st.info(
    "🧠 Model: Small CNN | Input: 128 × 128"
)


# =========================================================
# 7. IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# 8. PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------
    # Open image
    # -----------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------------
    # Display image
    # -----------------------------------------

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )


    # -----------------------------------------
    # Resize image
    # -----------------------------------------

    resized_image = image.resize(
        (128, 128)
    )


    # -----------------------------------------
    # Convert to NumPy
    # -----------------------------------------

    image_array = np.array(
        resized_image
    )


    # -----------------------------------------
    # Normalize
    # -----------------------------------------

    image_array = (
        image_array / 255.0
    )


    # -----------------------------------------
    # Add batch dimension
    # -----------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # =====================================================
    # 9. CNN PREDICTION
    # =====================================================

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]


    # =====================================================
    # 10. CLASSIFICATION
    # =====================================================

    if prediction >= 0.5:

        predicted_class = "dogs"

        confidence = prediction

    else:

        predicted_class = "cats"

        confidence = 1 - prediction


    confidence_value = (
        confidence * 100
    )


    # =====================================================
    # 11. DISPLAY RESULT
    # =====================================================

    st.success(
        f"Prediction: {predicted_class.upper()}"
    )


    st.metric(
        "Confidence",
        f"{confidence_value:.2f}%"
    )


    # =====================================================
    # 12. PREDICTION PROBABILITIES
    # =====================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )


    cat_probability = (
        (1 - prediction) * 100
    )

    dog_probability = (
        prediction * 100
    )


    # -----------------------------------------
    # Cat
    # -----------------------------------------

    st.write(
        f"🐱 Cats: {cat_probability:.2f}%"
    )

    st.progress(
        int(cat_probability)
    )


    # -----------------------------------------
    # Dog
    # -----------------------------------------

    st.write(
        f"🐶 Dogs: {dog_probability:.2f}%"
    )

    st.progress(
        int(dog_probability)
    )


    # =====================================================
    # 13. INTERPRETATION
    # =====================================================

    if confidence_value >= 90:

        st.success(
            "🎯 Very high confidence prediction!"
        )

    elif confidence_value >= 70:

        st.info(
            "👍 Good confidence prediction."
        )

    else:

        st.warning(
            "⚠️ The model is not very confident. "
            "Try a clearer image."
        )


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🧠 Small CNN • TensorFlow/Keras • Streamlit"
)
