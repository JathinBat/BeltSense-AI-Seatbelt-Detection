#!/usr/bin/env python3
"""
Simple HTTP server for YOLO model inference
Receives images from Flutter app and returns seatbelt classification results
"""

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

# Load the YOLO model once on startup
print("Loading YOLO model...")
model = YOLO('seatbelt_model/latest/weights/best.pt')
print("Model loaded successfully!")

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run YOLO inference
        results = model(img_array, verbose=False)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            # For classification model, get the top prediction
            if hasattr(result, 'probs') and result.probs is not None:
                # Get class probabilities
                probs = result.probs.data.cpu().numpy()
                
                # Assuming classes are [no_seatbelt, seatbelt]
                seatbelt_confidence = float(probs[1]) if len(probs) > 1 else 0.0
                no_seatbelt_confidence = float(probs[0]) if len(probs) > 0 else 0.0
                
                # Determine prediction
                has_seatbelt = seatbelt_confidence > no_seatbelt_confidence
                confidence = max(seatbelt_confidence, no_seatbelt_confidence)
                
                return jsonify({
                    'success': True,
                    'has_seatbelt': has_seatbelt,
                    'confidence': confidence,
                    'seatbelt_confidence': seatbelt_confidence,
                    'no_seatbelt_confidence': no_seatbelt_confidence,
                    'message': 'SEATBELT DETECTED - SAFE' if has_seatbelt else 'NO SEATBELT DETECTED - UNSAFE!'
                })
            else:
                return jsonify({'error': 'No classification results'}), 500
        else:
            return jsonify({'error': 'No results from model'}), 500
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': 'YOLO inference server is running'
    })

if __name__ == '__main__':
    print("Starting YOLO inference server...")
    print("Model path: seatbelt_model/latest/weights/best.pt")
    print("Server will run on http://localhost:5000")
    print("Endpoints:")
    print("  POST /classify - Classify seatbelt image")
    print("  GET /health - Health check")
    
    app.run(host='0.0.0.0', port=5000, debug=False)