# 🤖 AI-Powered Navigation Aid for the Visually Impaired

A real-time, AI-powered obstacle avoidance system designed to assist visually impaired individuals by providing audible navigation commands. This project uses a standard webcam, the YOLOv5 object detection model, and a custom Python script to identify and communicate potential hazards.

---

## 🚀 Key Features

* **Real-Time Object Detection:** Utilizes a YOLOv5 model to detect 15+ common obstacles (people, cars, chairs, etc.) from a live webcam feed.
* **Spatial Awareness Logic:** Divides the camera's view into three zones (Left, Center, Right) to determine the obstacle's location.
* **Proximity-Based Alerts (Depth Proxy):** Uses the bounding box *area* of a detected object as a proxy for distance, allowing it to prioritize the closest, most immediate threats.
* **Clear Voice Commands:** Provides simple, direct, and priority-based audible alerts (e.g., "Warning! Stop. Object very close," "Obstacle on the left. Please go right," or "Path is clear.") using `pyttsx3`.
* **Highly Tunable:** All critical logic (confidence, object types, distance thresholds) is stored in easy-to-edit configuration variables.

---

## 🛠️ How It Works

1.  **Capture:** The script captures the webcam feed using **OpenCV**.
2.  **Detect:** Each frame is passed to a pre-trained **YOLOv5** model via **PyTorch**.
3.  **Filter:** Detections are filtered based on a `CONFIDENCE_THRESHOLD` and a `CRITICAL_OBJECTS` list, ignoring irrelevant items.
4.  **Analyze:** For each critical obstacle, the script calculates two things:
    * **Zone:** The horizontal center of the bounding box (Left, Center, or Right).
    * **Proximity:** The *area* of the bounding box (a larger area means a closer object).
5.  **Prioritize:** The list of obstacles is sorted by proximity, so the system *only* alerts the user about the most immediate danger.
6.  **Alert:** Based on the primary obstacle's proximity and zone, a specific, helpful command is generated and spoken. A cooldown prevents repetitive alerts.

---

## 💻 Technologies Used

* **Python 3**
* **PyTorch:** For running the YOLOv5 model.
* **OpenCV:** For video capture and image processing.
* **YOLOv5s (by Ultralytics):** For real-time object detection.
* **pyttsx3:** For text-to-speech voice alerts.

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/balachandran-07/AI-powered-smart-glass.git
    cd AI-powered-smart-glass
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python smartglass.py
    ```
    Press 'q' to exit the application.

---

## 📈 Future Improvements

* **Hardware Implementation:** Porting the project to a wearable device like a **Raspberry Pi** with a camera module and bone-conduction headphones.
* **True Depth Perception:** Integrating a stereo camera (like an **OAK-D Lite**) to get real depth data (in meters) instead of relying on a bounding box proxy.
* **Text Recognition (OCR):** Adding text recognition to read signs (e.g., "STOP," "EXIT").
