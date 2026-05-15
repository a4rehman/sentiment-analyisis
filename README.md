# 🧠 EmotiSense AI: Real-time Emotion Intelligence System

EmotiSense AI is a comprehensive, production-ready emotion detection and analytics platform. Built with Python, OpenCV, and Deep Learning (FER/DeepFace), it provides real-time facial expression analysis from webcams, video files, and RTSP streams.

![Emotion Detection Demo](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face_detection_opencv.jpg) *(Example Illustration)*

## 🚀 Features

- **Multi-Source Support**: Real-time webcam, uploaded video files, and CCTV/IP camera (RTSP) streams.
- **Dual Model Engine**:
  - **FER**: Lightweight, high-speed detection (best for real-time).
  - **DeepFace**: High-accuracy analysis with multiple backends (SSD, MTCNN, RetinaFace).
- **Advanced Tracking**: Persistent face tracking across frames using IoU-based centroid assignment.
- **Intelligent Alerts**: Real-time warnings for high anger levels and crowd mood shifts.
- **Analytics Dashboard**: Dynamic charts for emotion distribution, confidence trends, and face counts.
- **Automated Logging**: Every detection event is saved to thread-safe CSV logs.
- **Report Generation**: Export session summaries (TXT) and combined logs (CSV).
- **Modern UI**: Sleek, dark-themed Streamlit interface with responsive design.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotisense-ai.git
cd emotisense-ai
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
sentiment-analysis/
├── assets/             # UI assets and CSS
│   └── style.css       # Custom dark theme
├── data/               # Persistent storage
│   ├── logs/           # Session CSV logs
│   └── snapshots/      # Face snapshots (optional)
├── modules/            # Core logic
│   ├── alerts.py       # Alert system
│   ├── analytics.py    # Charting & stats
│   ├── detector.py     # Emotion detection (FER/DeepFace)
│   ├── logger.py       # CSV logging logic
│   ├── tracker.py      # Face tracking
│   └── video_processor.py # OpenCV pipeline
├── utils/              # Helper utilities
│   ├── drawing.py      # Annotation logic
│   └── helpers.py      # File & frame utils
├── app.py              # Main Streamlit frontend
├── config.py           # Central configuration
└── requirements.txt    # Project dependencies
```

---

## 📊 Analytics & Metrics

The system tracks seven core emotions:
- **😊 Happy** | **😢 Sad** | **😡 Angry** | **😐 Neutral**
- **😨 Fear** | **😲 Surprise** | **🤢 Disgust**

### Advanced Capabilities
- **Emotion Trend Graph**: Visualizes how confidence levels change over time.
- **Crowd Mood Analysis**: Aggregates individual emotions to determine the overall vibe.
- **Alert System**: Triggers "Critical" alerts when the number of angry faces exceeds user-defined thresholds.

---

## 🔧 Deployment Steps

### Streamlit Cloud
1. Push your code to a GitHub repository.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your repo and select `app.py` as the main entry point.
4. Add any necessary environment variables in the Streamlit Dashboard.

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🔮 Future Improvements

- [ ] **Face Recognition**: Integrate with a database to identify specific individuals.
- [ ] **Gaze Tracking**: Analyze where users are looking for heatmaps.
- [ ] **Cloud DB Integration**: Sync logs to MongoDB or PostgreSQL for long-term storage.
- [ ] **Model Quantization**: Optimize DeepFace models for Edge devices (Raspberry Pi/Jetson).
- [ ] **Audio-Visual Sentiment**: Combine facial expressions with voice tone analysis.

---

## 📄 License
MIT License - See [LICENSE](LICENSE) for details.
