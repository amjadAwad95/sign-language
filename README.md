# Arabic Sign Language Detection 🤟

An AI-powered real-time Arabic Sign Language detection system using YOLOv11. This project can detect Arabic sign language gestures through a camera feed and provide both visual feedback and Arabic audio pronunciation of the detected words.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-orange.svg)

## 🌟 Features

- **Real-time Detection**: Live camera feed processing for sign language recognition
- **YOLOv11 Model**: State-of-the-art object detection for accurate gesture recognition
- **Arabic Translation**: Converts detected English sign words to Arabic text
- **Audio Feedback**: Text-to-speech functionality in Arabic for detected signs
- **Web Interface**: User-friendly Gradio web interface for easy interaction
- **Dual Mode**: Both terminal-based (`app.py`) and web-based (`main.py`) applications

## 🎯 Supported Signs

The model currently recognizes 13 common Arabic sign language gestures:

| English | Arabic | English | Arabic |
|---------|--------|---------|--------|
| Hello | مرحبا | Dog | كلب |
| Thanks | شكرا | Love | حب |
| Yes | نعم | Me | أنا |
| No | لا | You | أنت |
| Sorry | آسف | Mother | أم |
| Fine | بخير | Smile | ابتسامة |
| Sunday | الأحد | | |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- CUDA-capable GPU (optional, for better performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/arabic-sign-language.git
   cd arabic-sign-language
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   **Option A: Web Interface (Recommended)**
   ```bash
   python main.py
   ```
   Then open your browser to the displayed local URL (usually `http://127.0.0.1:7860`)

   **Option B: Terminal Interface**
   ```bash
   python app.py
   ```
   Press 'q' to quit the camera feed.

## 🏗️ Project Structure

```
arabic-sign-language/
├── app.py                          # Terminal-based detection app
├── main.py                         # Gradio web interface app
├── requirements.txt                # Python dependencies
├── sign_language_model_train.ipynb # Model training notebook
├── test.py                         # Testing utilities
├── test.ipynb                      # Testing notebook
├── README.md                       # Project documentation
├── model/                          # Trained models
│   ├── model.onnx                  # ONNX format model (production)
│   └── model.pt                    # PyTorch format model
├── audio/                          # Generated audio files
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── audio.py                    # Text-to-speech functionality
│   ├── detection.py                # Detection logic
│   └── mappings.py                 # English-Arabic translations
└── data.yaml                       # YOLO training configuration
```

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: YOLOv11s (small variant for balance of speed and accuracy)
- **Framework**: Ultralytics YOLO
- **Input Size**: 640x640 pixels
- **Format**: ONNX (optimized for deployment)

### Performance Optimizations
- ONNX Runtime for faster inference
- CUDA acceleration when available
- Frame skipping for real-time performance
- Audio cooldown to prevent spam

### Dependencies
- **ultralytics**: YOLO model implementation
- **torch & torchvision**: PyTorch framework
- **opencv-python**: Computer vision operations
- **gradio**: Web interface framework
- **gTTS**: Google Text-to-Speech
- **pygame**: Audio playback
- **onnxruntime-gpu**: ONNX inference optimization

## 📚 Usage Examples

### Web Interface
1. Launch the web app: `python main.py`
2. Allow camera permissions in your browser
3. Point your camera at sign language gestures
4. Watch real-time detection with Arabic translation and audio

### Terminal Interface
1. Run: `python app.py`
2. Position yourself in front of the camera
3. Perform sign language gestures
4. Listen for Arabic pronunciation of detected signs
5. Press 'q' to quit

## 🎓 Model Training

The model was trained using YOLOv11 with custom Arabic sign language dataset:

```python
yolo detect train model=yolo11s.pt data=data.yaml epochs=60 imgsz=640 project="arabic-sl-yolov11" name="arabic-sl"
```

Training details:
- **Epochs**: 60
- **Image Size**: 640x640
- **Base Model**: YOLOv11s pre-trained weights
- **Tracking**: Weights & Biases integration

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- 📈 Adding more sign language gestures
- 🌍 Supporting other Arabic dialects
- ⚡ Performance optimizations
- 🎨 UI/UX improvements
- 📱 Mobile app development

## 📊 Performance

- **Real-time Processing**: 15-30 FPS depending on hardware
- **Accuracy**: Trained on diverse Arabic sign language dataset
- **Latency**: < 100ms inference time with GPU acceleration
- **Memory Usage**: ~2GB with CUDA, ~1GB CPU-only

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Ensure camera permissions are granted
   - Try changing camera index in code (0, 1, 2...)

2. **Slow performance**
   - Install CUDA version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - Reduce inference frequency in config

3. **Audio not working**
   - Check system audio settings
   - Install/update pygame: `pip install --upgrade pygame`

4. **Model not found**
   - Ensure `model/model.onnx` exists
   - Download pre-trained model or train your own

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLO framework
- **Arabic Sign Language** community for datasets and guidance
- **Google Text-to-Speech** for Arabic audio generation
- **Gradio** team for the excellent web interface framework

## 📞 Contact

- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/your-profile)

---

Made with ❤️ for the Arabic Sign Language community