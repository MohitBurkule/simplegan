import streamlit as st
import tensorflow as tf
from simplegan.gan import DCGAN
from PIL import Image
import numpy as np

# Ensure the GAN runs fast
@st.cache_resource
def load_model():
    # Initialize the model and possibly load trained weights if available
    gan = DCGAN()
    # Mocking image size for fast testing
    gan.image_size = (64, 64, 3)
    # Initialize internal models
    gan.gen_model = gan.generator()
    return gan

gan = load_model()

st.title("SimpleGAN Image Generator")

st.write("This app uses a pre-initialized DCGAN model to generate images. Training can take a long time, so for a fast app, it relies on inference or minimal overhead.")

n_samples = st.slider("Number of samples to generate", 1, 10, 5)

if st.button("Generate Images"):
    with st.spinner("Generating..."):
        # We need to make sure generate_samples works since we mocked stuff.
        # But we'll try and see. If gen_model is not None, generate_samples uses it.
        generated_samples = gan.generate_samples(n_samples=n_samples)

        for i, sample in enumerate(generated_samples):
            # Normalize to 0-255 if it's in a different range, e.g., -1 to 1
            if np.min(sample) < 0:
                sample = (sample + 1.0) / 2.0

            # SimpleGAN samples might need scaling to 255
            sample_img = np.clip(sample * 255, 0, 255).astype(np.uint8)

            st.image(sample_img, caption=f"Sample {i+1}")
