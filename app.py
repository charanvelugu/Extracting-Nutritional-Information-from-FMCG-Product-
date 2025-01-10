import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def process_image(image):
    # Open the uploaded image using PIL
    image = Image.open(image)

    # Convert to RGB (PaddleOCR expects RGB format)
    image_rgb = image.convert('RGB')

    # Convert to numpy array (PaddleOCR works with numpy arrays)
    image_rgb_np = np.array(image_rgb)

    # Use PaddleOCR to extract text
    result = ocr.ocr(image_rgb_np, cls=True)

    # Extract and format the detected text
    text = ""
    for line in result[0]:
        text += line[1][0]+"\n"
    
    return text

def main():
    st.title("Image Text Recognition with PaddleOCR and PIL")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Process the image to extract text
        extracted_text = process_image(uploaded_image)

        # Display the extracted text
        if extracted_text:
            st.subheader("Extracted Text:")
            st.text_area("", extracted_text, height=200)
        else:
            st.warning("No text detected.")

if __name__ == "__main__":
    main()
