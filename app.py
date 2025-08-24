import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Model configuration
VIDEO_LENGTH = 20
FRAME_SIZE = (112, 112)
CLASS_NAMES = [
    "Fighting", "Shoplifting", "Vandalism", "Assault", 
    "Burglary", "Robbery", "Abduction", "Loitering",
    "Trespassing", "Arson", "Harassment", "DrugDealing",
    "RecklessDriving", "PublicIntoxication", "SuspiciousObject"
]

# Load pre-trained model
model = load_model('model/suspicious_activity_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = frame / 255.0
        frames.append(frame)
        
        if len(frames) == VIDEO_LENGTH:
            break
            
    cap.release()
    
    # Pad with black frames if needed
    while len(frames) < VIDEO_LENGTH:
        frames.append(np.zeros((*FRAME_SIZE, 3)))
            
    return np.expand_dims(np.array(frames), axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess and predict
        processed_video = preprocess_video(filepath)
        prediction = model.predict(processed_video)
        pred_index = np.argmax(prediction[0])
        confidence = float(prediction[0][pred_index])
        activity = CLASS_NAMES[pred_index]
        
        # Generate confidence chart data
        confidences = {CLASS_NAMES[i]: float(pred) for i, pred in enumerate(prediction[0])}
        
        return render_template('result.html', 
                               video_file=file.filename,
                               activity=activity,
                               confidence=confidence,
                               confidences=confidences)
    
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
