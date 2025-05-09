import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2

# Set page config with light theme
st.set_page_config(
    page_title="IdentiFy",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    h1 {
        color: #2E4057;
        font-size: 2.2rem;
    }
    .stButton button {
        background-color: #4F9EDE;
        color: white;
        border-radius: 5px;
    }
    .css-1kyxreq {
        justify-content: center;
    }
    .prediction-box {
        color: #013863;
        background-color: #b4dbfa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 style='text-align: center;'>IdentiFy</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666;'>Simple Gender Prediction from Photos</p>", unsafe_allow_html=True)

# Dictionary for gender mapping
gender_dict = {0: "Female", 1: "Male"}

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(img: Image.Image, padding: int = 50):
    # Convert PIL image to OpenCV format
    img_cv = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None  # No face detected

    # Use the first detected face
    (x, y, w, h) = faces[0]
    
    # Add padding and keep within image bounds
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.width)
    y2 = min(y + h + padding, img.height)
    
    # Crop the face with padding
    face = img.crop((x1, y1, x2, y2))
    return face

# Function to preprocess the image
def preprocess_image(img):
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 128x128 pixels
    img = img.resize((128, 128), Image.LANCZOS)
    # Convert to numpy array
    img_array = np.array(img)
    # Normalize to 0-1
    img_array = img_array / 255.0
    # Reshape for model input
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

# Function to make predictions
def predict_gender(img_array, model):
    pred = model.predict(img_array)
    pred_gender = gender_dict[int(round(pred[0][0][0]))]
    return pred_gender

# Load the model directly from file
@st.cache_resource
def load_model_once():
    try:
        # Define custom objects if needed for your model
        custom_objects = {}
        # Specify the path to your model file
        model_path = "model.keras"
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Compile the model with appropriate loss functions and metrics
        model.compile(
            loss={
                'gender_output': 'binary_crossentropy',
                'age_output': 'mae'
            },
            optimizer='adam',
            metrics={
                'gender_output': ['accuracy'],
                'age_output': ['mae']
            }
        )
        
        return model, None
    except Exception as e:
        return None, str(e)

# Load model
model, error = load_model_once()

if error:
    st.error(f"Error loading model: {error}")
else:
    # Create a centered container with max-width
    container = st.container()
    
    with container:
        # File uploader with cleaner design
        uploaded_file = st.file_uploader("Upload a photo with a clear face", type=["jpg", "jpeg", "png"], 
                                        label_visibility="visible", key="file_uploader")
        
        col1, col2 = st.columns([1, 1])
        
        if uploaded_file is not None:
            # Display the image
            image = Image.open(uploaded_file)
            
            with col1:
                st.image(image, use_container_width=True, caption="Uploaded Image")
            
            # Process and predict
            with st.spinner("Analyzing..."):
                # Detect and crop face
                cropped_face = detect_and_crop_face(image)
                
                if cropped_face is None:
                    st.warning("No face detected. Please upload a clearer image.")
                else:
                    with col2:
                        st.image(cropped_face, use_container_width=True, caption="Detected Face")
                    
                    # Preprocess the image and predict
                    processed_img = preprocess_image(cropped_face)
                    pred_gender = predict_gender(processed_img, model)
                    
                    # Display result with nice formatting
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>Result</h2>
                        <h3>Gender: {pred_gender}</h3>
                    </div>
                    """, unsafe_allow_html=True)

# Minimal footer
st.markdown("<hr style='margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #999999; font-size: 0.8rem;'>IdentiFy • Simple Gender Prediction</p>", unsafe_allow_html=True)