#!/usr/bin/env python3
"""
Convert YOLOv8 PyTorch model to TensorFlow Lite format for Flutter app
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def convert_yolo_to_tflite():
    """Convert the latest YOLO model to TensorFlow Lite format"""
    
    # Get the latest model
    model_path = Path("seatbelt_model/latest/weights/best.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"🔄 Converting YOLO model to TensorFlow Lite...")
    print(f"📂 Source: {model_path}")
    
    try:
        # Load the YOLO model
        model = YOLO(str(model_path))
        
        # Export to TensorFlow Lite format
        # This creates a .tflite file in the same directory
        tflite_path = model.export(format='tflite', imgsz=640)
        
        print(f"✅ Conversion successful!")
        print(f"📂 TensorFlow Lite model: {tflite_path}")
        
        # Copy to Flutter assets
        flutter_assets = Path("Flutter APp/seatbelt_detector_app/assets/models")
        flutter_assets.mkdir(parents=True, exist_ok=True)
        
        import shutil
        dest_path = flutter_assets / "seatbelt_model.tflite"
        shutil.copy2(tflite_path, dest_path)
        
        print(f"📱 Copied to Flutter assets: {dest_path}")
        print(f"📊 Model ready for mobile inference!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    success = convert_yolo_to_tflite()
    sys.exit(0 if success else 1)