# SafeZone: Suspicious Activity Detection

SafeZone is a web-based application designed to analyze video streams and detect suspicious activities using a deep learning model. Users can upload a video file, and the system will process it to identify and classify potential threats from a predefined list of activities like "Fighting", "Shoplifting", "Vandalism", etc.

![Demo Screenshot Placeholder](https://via.placeholder.com/800x450.png?text=App+Screenshot+Here)
*(Add a screenshot or GIF of your application in action here)*

---

## Features

- **Video Upload**: Simple web interface to upload video files (MP4, AVI, MOV).
- **AI-Powered Analysis**: Utilizes a TensorFlow/Keras model to classify activities in the video.
- **Immediate Feedback**: Displays the most likely suspicious activity detected.
- **Confidence Score**: Shows the model's confidence in its prediction.
- **Detailed Breakdown**: Provides a chart visualizing the confidence scores for all possible activity classes.
- **Ready for Deployment**: Includes `gunicorn` and a `Procfile` for easy deployment on platforms like Heroku.

---

## Technical Stack

- **Backend**: Flask
- **Machine Learning**: TensorFlow 2, Keras
- **Video Processing**: OpenCV
- **Numerical Operations**: NumPy
- **Frontend**: HTML5, CSS3, JavaScript (with Chart.js)
- **WSGI Server**: Gunicorn

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.8+
- `pip` and `venv`

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SafeZone-Suspicious-Activity-Detection.git
    cd SafeZone-Suspicious-Activity-Detection
    ```

2.  **Create and activate a virtual environment:**
    - On macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add the Trained Model:**
    The core of this application is the pre-trained deep learning model.
    - Create a directory named `model`.
    - Place your trained Keras model file inside it and name it `suspicious_activity_model.h5`.

    The final structure should be: `model/suspicious_activity_model.h5`.

---

## How to Run the Application

### 1. Development Mode (for testing)

Use the built-in Flask development server.

```bash
python app.py

How to Use the App
Open your web browser and navigate to http://127.0.0.1:5000.
Click the "Choose File" button to select a video file (.mp4, .avi, .mov).
Click the "Upload & Analyze" button.
Wait for the model to process the video.
The results page will display the uploaded video, the detected activity, the confidence score, and a bar chart with the prediction breakdown.
Project Configuration
Key parameters can be configured at the top of the app.py file:
UPLOAD_FOLDER: Directory to store uploaded videos.
MAX_CONTENT_LENGTH: Maximum allowed file size for uploads.
ALLOWED_EXTENSIONS: Permitted video file extensions.
VIDEO_LENGTH: Number of frames to sample from the video.
FRAME_SIZE: The target size (height, width) for each frame.
CLASS_NAMES: The list of activity classes the model can predict. This must match the order used during model training.
License
This project is licensed under the MIT License. See the LICENSE file for details.
code
Code
#### 2. `requirements.txt`

(As provided by you)