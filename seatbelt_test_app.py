"""
Seatbelt Detection Test App
A simple web interface to test your trained seatbelt detection model
"""

import streamlit as st
import os
from pathlib import Path
from PIL import Image
import io
from model_manager import get_latest_model_path

# Configure page
st.set_page_config(
    page_title="Seatbelt Detection Tester",
    page_icon=None,
    layout="wide"
)

def find_model_weights():
    """Find the latest trained model weights - prioritize latest folder"""
    base_path = Path(__file__).parent
    
    # First priority: Check for latest model folder
    latest_model_path = base_path / "seatbelt_model" / "latest" / "weights" / "best.pt"
    if latest_model_path.exists():
        return str(latest_model_path)
    
    # Second priority: Look for best.pt files in any folder
    best_models = list(base_path.glob("seatbelt_model/*/weights/best.pt"))
    
    if best_models:
        # Sort by modification time and return the most recent best.pt
        latest_model = sorted(best_models, key=lambda x: x.stat().st_mtime)[-1]
        return str(latest_model)
    
    # Fallback to last.pt files if no best.pt found
    last_models = list(base_path.glob("seatbelt_model/*/weights/last.pt"))
    
    if last_models:
        latest_model = sorted(last_models, key=lambda x: x.stat().st_mtime)[-1]
        return str(latest_model)
    
    return None

def load_model(model_path):
    """Load the YOLO model"""
    try:
        import importlib

        # Dynamically import to be resilient to package layout and avoid
        # static import-time linter errors on different environments.
        ultralytics_mod = importlib.import_module('ultralytics')
        YOLO = getattr(ultralytics_mod, 'YOLO', None)
        if YOLO is None:
            try:
                yolo_mod = importlib.import_module('ultralytics.yolo')
                YOLO = getattr(yolo_mod, 'YOLO')
            except Exception:
                raise

        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def classify_image(model, image):
    """Classify an uploaded image"""
    try:
        # Convert PIL image to bytes for YOLO
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Run prediction
        results = model(image)
        result = results[0]
        
        # Get prediction details
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf.item()
        
        # Get all class probabilities
        probs = result.probs.data.cpu().numpy()
        class_names = [result.names[i] for i in range(len(probs))]
        
        # Convert probabilities to Python floats
        prob_dict = {name: float(prob) for name, prob in zip(class_names, probs)}
        
        return {
            'prediction': class_name,
            'confidence': confidence,
            'all_probs': prob_dict
        }
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

def main():
    st.title("Seatbelt Detection Test App")
    st.markdown("Upload an image to test your trained seatbelt detection model")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Auto-find model or allow manual path
    # Prefer explicit latest model published by model_manager
    auto_model_path = get_latest_model_path() or find_model_weights()

    if auto_model_path:
        model_dir = Path(auto_model_path).parent.parent.name
        model_file = Path(auto_model_path).name
        model_time = Path(auto_model_path).stat().st_mtime
        from datetime import datetime
        model_date = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M")

        st.sidebar.success("Latest trained model found")
        st.sidebar.info(f"{model_dir}")
        st.sidebar.info(f"{model_file}")
        st.sidebar.info(f"{model_date}")
        use_auto = st.sidebar.checkbox("Use latest model", value=True)
    else:
        st.sidebar.warning("No trained model found")
        use_auto = False
    
    # Model path selection
    if use_auto and auto_model_path:
        model_path = auto_model_path
    else:
        model_path = st.sidebar.text_input(
            "Model path (best.pt file):",
            placeholder="seatbelt_model/dynamic_training/weights/best.pt"
        )
    
    # Load model
    if model_path and os.path.exists(model_path):
        if 'model' not in st.session_state or st.session_state.get('model_path') != model_path:
            with st.spinner("Loading model..."):
                model = load_model(model_path)
                if model:
                    st.session_state.model = model
                    st.session_state.model_path = model_path
                    st.sidebar.success("Model loaded successfully!")
                else:
                    st.sidebar.error("Failed to load model")
                    return
    elif model_path:
        st.sidebar.error("Model file not found!")
        return
    else:
        st.info("👆 Please select a model in the sidebar to get started")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to test seatbelt detection"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add image info
            st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.header("Prediction Results")
        
        if uploaded_file is not None and 'model' in st.session_state:
            with st.spinner("Analyzing image..."):
                result = classify_image(st.session_state.model, image)
            
            if result:
                # Main prediction
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Color code the result
                if 'seatbelt' in prediction.lower() and 'no' not in prediction.lower():
                    st.success("SEATBELT DETECTED")
                    st.success(f"Prediction: {prediction}")
                    st.success(f"Confidence: {confidence:.1%}")
                else:
                    st.error("NO SEATBELT DETECTED")
                    st.error(f"Prediction: {prediction}")
                    st.error(f"Confidence: {confidence:.1%}")
                
                # Detailed probabilities
                st.subheader("Detailed Probabilities")
                for class_name, prob in result['all_probs'].items():
                    # Convert to float and ensure it's between 0 and 1
                    prob_value = float(prob)
                    percentage = prob_value * 100
                    st.write(f"**{class_name}:** {percentage:.1f}%")
                    st.progress(prob_value)
                
                # Confidence interpretation
                st.subheader("Confidence Level")
                if confidence > 0.9:
                    st.info("Very High Confidence")
                elif confidence > 0.7:
                    st.info("High Confidence")
                elif confidence > 0.5:
                    st.warning("Moderate Confidence")
                else:
                    st.warning("Low Confidence - Consider manual review")
        
        elif uploaded_file is None:
            st.info("Upload an image to see results")
    
    # Usage instructions
    st.markdown("---")
    st.subheader("How to Use")
    st.markdown("""
     1. **Select Model**: The app will auto-detect your trained model, or you can specify a path manually
     2. **Upload Image**: Click "Browse files" to upload a JPG, PNG, or JPEG image
     3. **View Results**: The prediction and confidence will appear on the right
     4. **Interpret Results**: 
         - Green = Seatbelt detected (safe)
         - Red = No seatbelt detected (unsafe)
         - Higher confidence = more reliable prediction
    """)
    
    # Model training info
    if 'model' in st.session_state:
        st.markdown("---")
        st.subheader("Model Information")
        try:
            model_info = st.session_state.model.info()
            st.text(f"Model: {Path(st.session_state.model_path).parent.parent.name}")
            st.text(f"Weights: {Path(st.session_state.model_path).name}")
        except:
            st.text(f"Using model: {Path(st.session_state.model_path).name}")

if __name__ == "__main__":
    main()