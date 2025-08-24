# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten,
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import cv2
import numpy as np

# Configuration
VIDEO_LENGTH = 20  # Number of frames per video clip
FRAME_SIZE = (112, 112)  # Resized frame dimensions
BATCH_SIZE = 8
EPOCHS = 30
CLASSES = 15  # Number of activity classes

# Data Generator
class VideoFrameGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, target_size, n_frames, n_classes, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_paths = self.video_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.empty((self.batch_size, self.n_frames, *self.target_size, 3))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            frames = self._load_video_frames(path)
            X[i,] = frames
            y[i] = label
            
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.target_size)
            frame = frame / 255.0  # Normalization
            frames.append(frame)
            
            if len(frames) == self.n_frames:
                break
                
        cap.release()
        
        # Pad with black frames if video is shorter than n_frames
        while len(frames) < self.n_frames:
            frames.append(np.zeros((*self.target_size, 3)))
            
        return np.array(frames)
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.video_paths))
            np.random.shuffle(indices)
            self.video_paths = self.video_paths[indices]
            self.labels = self.labels[indices]

# Model Architecture
def create_model(input_shape, n_classes):
    model = Sequential([
        # Spatial feature extraction
        TimeDistributed(Conv2D(32, (3,3), activation='relu'), 
                       input_shape=input_shape),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2,2))),
        
        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
        TimeDistributed(Dropout(0.25)),
        TimeDistributed(MaxPooling2D((2,2))),
        
        TimeDistributed(Conv2D(128, (3,3), activation='relu')),
        TimeDistributed(Flatten()),
        
        # Temporal processing
        LSTM(256, return_sequences=True),
        LSTM(128),
        
        # Classification
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Dataset Preparation
def prepare_dataset(dataset_path):
    classes = sorted(os.listdir(dataset_path))
    video_paths = []
    labels = []
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        for video_file in os.listdir(class_dir):
            video_paths.append(os.path.join(class_dir, video_file))
            labels.append(label)
            
    return np.array(video_paths), np.array(labels)

# Main Execution
if __name__ == "__main__":
    # Dataset paths
    train_path = "Dataset/TinyVIRAT/videos/train"
    test_path = "Dataset/TinyVIRAT/videos/test"
    
    # Prepare datasets
    train_videos, train_labels = prepare_dataset(train_path)
    test_videos, test_labels = prepare_dataset(test_path)
    
    # Create generators
    train_gen = VideoFrameGenerator(
        train_videos, train_labels,
        batch_size=BATCH_SIZE,
        target_size=FRAME_SIZE,
        n_frames=VIDEO_LENGTH,
        n_classes=CLASSES
    )
    
    test_gen = VideoFrameGenerator(
        test_videos, test_labels,
        batch_size=BATCH_SIZE,
        target_size=FRAME_SIZE,
        n_frames=VIDEO_LENGTH,
        n_classes=CLASSES,
        shuffle=False
    )
    
    # Create model
    model = create_model(
        input_shape=(VIDEO_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1], 3),
        n_classes=CLASSES
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        "activity_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Training
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    
    # Evaluation
    model.evaluate(test_gen)
    
    # Save model
    model.save("suspicious_activity_model.h5")
