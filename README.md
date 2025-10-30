# BeltSense - AI-Powered Seatbelt Detection System

![BeltSense Logo](newAppCover.png)

BeltSense is a comprehensive real-time seatbelt detection system that combines advanced computer vision, artificial intelligence, and mobile application development. The system features a custom-trained YOLOv8 classification model integrated with a Flutter mobile application for seamless, real-time seatbelt detection.

## 📊 Model Performance
- **Accuracy**: 95.7% (achieved at epoch 17)
- **Training Images**: 1,710 (767 seatbelt, 943 no-seatbelt)
- **Validation Images**: 370 (133 seatbelt, 237 no-seatbelt)
- **Total Dataset**: 2,080 images

## 🎯 Usage

### Train the Model
```bash
python run_training.py
```

### Real-time Webcam Detection 📹
```bash
python webcam_detector.py
```

### Test Random Images
```bash
python runModel.py
```

### Classify Static Images
```python
from seatbelt_classifier import classify_image, classify_folder

# Classify a single image
classify_image("path/to/your/image.jpg")

# Classify all images in a folder
classify_folder("path/to/your/folder")
```

## 📁 Project Structure
```
├── run_training.py          # Main training script
├── seatbelt_classifier.py   # Image classification functions
├── runModel.py              # Random image testing
├── webcam_detector.py       # 📹 Real-time webcam detection
├── test_webcam.py           # Webcam functionality test
├── seatbelt_dataset/        # Organized training data
├── seatbelt_model/          # Trained model outputs
│   └── final/weights/best.pt # Best trained model
└── requirements.txt         # Dependencies
```

## 🔧 Requirements
- Python 3.8+
- ultralytics (YOLOv8)
- opencv-python (for webcam)

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start
1. **Train the model**: `python run_training.py`
2. **Test webcam**: `python test_webcam.py` (optional)
3. **Real-time detection**: `python webcam_detector.py`
4. **Test with random images**: `python runModel.py`

## 📹 Webcam Features
- Real-time seatbelt detection
- Live confidence scoring
- Mirror mode display
- Screenshot capture (press 's')
- Visual safety indicators
- Quit with 'q' key

The model will automatically detect seatbelts with high accuracy and provide confidence scores for each classification.

## 📱 BeltSense Mobile App

The Flutter mobile application provides a professional, user-friendly interface for real-time seatbelt detection:

### Features:
- **Real-time Detection**: Live camera feed with instant seatbelt detection
- **ONNX Integration**: Optimized model inference using ONNX Runtime
- **Audio Alerts**: Automatic alerts when no seatbelt is detected
- **Cross-Platform**: Works on Android, iOS, and desktop platforms
- **Professional UI**: Clean, modern interface with status indicators

### Mobile App Structure:
```
Flutter APp/seatbelt_detector_app/
├── lib/main.dart                     # Main app logic with ONNX integration
├── android/                          # Android-specific configuration
│   └── app/src/main/kotlin/         # Native ONNX inference handler
├── assets/models/best.onnx          # ONNX model (5.7MB)
├── assets/audio/                    # Alert sound files
└── assets/images/app_icon.png       # App icon (newAppCover.png)
```

### Getting Started with Mobile App:
```bash
cd "Flutter APp/seatbelt_detector_app"
flutter pub get
flutter build apk  # For Android
flutter run         # For development
```

## 🔧 Technical Implementation

### AI Model Pipeline:
1. **YOLOv8n-cls** trained on custom seatbelt dataset
2. **PyTorch (.pt)** → **ONNX (.onnx)** conversion for mobile deployment
3. **Real-time inference** with ~50-100ms latency on mobile devices
4. **Input**: 224x224 RGB images
5. **Output**: Binary classification (seatbelt/no_seatbelt) with confidence scores

### Mobile Architecture:
- **Flutter Framework**: Cross-platform UI development
- **Method Channels**: Native Android integration
- **ONNX Runtime**: Optimized AI inference engine
- **Camera Integration**: Real-time image capture and processing

## 🎯 Use Cases

- **Automotive Safety**: Vehicle system integration
- **Fleet Management**: Driver safety monitoring
- **Insurance Applications**: Risk assessment and compliance
- **Educational Tools**: Safety awareness applications
- **Research Platform**: Computer vision and AI research

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- iOS native ONNX integration
- Additional model architectures
- Enhanced UI/UX features
- Performance optimizations

## 📄 License

This project is licensed under the MIT License.

---

**BeltSense** - Making roads safer, one detection at a time. 🚗✨
