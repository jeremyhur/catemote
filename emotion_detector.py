import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import threading
import time
from fer import FER

class EmotionDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Detection App")
        self.root.geometry("2200x1100")

        # --- Webcam Initialization ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

        # --- Model Initialization ---
        self.yolo_model = self.load_yolo_model()
        self.fer_detector = self.load_fer_detector()

        # --- Emotion to Image Mapping ---
        self.emotion_images = {
            'angry': 'emotion_images/angry.png',
            'disgust': 'emotion_images/angry.png',
            'fear': 'emotion_images/surprise.png',
            'happy': 'emotion_images/happy.png',
            'sad': 'emotion_images/sad.png',
            'surprise': 'emotion_images/surprise.png',
            'neutral': 'emotion_images/sad.png'
        }
        self.current_emotion = 'neutral'
        self.last_stable_emotion_data = None
        self.last_emotion_update_time = 0

        # --- Threading and Frame Management for Performance ---
        # A lock is used to prevent race conditions when accessing shared frames
        # between the main GUI thread and the background processing thread.
        self.frame_lock = threading.Lock()
        self.latest_frame = None      # Stores the latest raw frame from the webcam
        self.processed_frame = None   # Stores the latest frame with all overlays drawn
        self.running = True

        # --- GUI Setup ---
        self.setup_gui()

        # --- Start Background Processing Thread ---
        # This single thread will handle all heavy tasks (face detection, emotion analysis)
        self.processing_thread = threading.Thread(target=self.video_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # --- Start Webcam Update Loop ---
        self.update_webcam()

    def load_yolo_model(self):
        """Loads the YOLOv8 face detection model."""
        try:
            print("Downloading YOLOv8 Face Detection model...")
            model_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection",
                filename="model.pt"
            )
            print("YOLOv8 Face Detection model loaded successfully.")
            return YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLOv8 Face Detection model: {e}")
            print("Falling back to general YOLOv8 model...")
            try:
                return YOLO('yolov8n.pt')
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                return None

    def load_fer_detector(self):
        """Initializes the FER model for emotion recognition."""
        try:
            print("Initializing FER for enhanced sensitivity...")
            return FER(mtcnn=True)
        except Exception as e:
            print(f"FER initialization failed: {e}")
            return None

    def setup_gui(self):
        """Creates the main graphical user interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Webcam
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        ttk.Label(left_frame, text="Live Webcam Feed", font=('Calibri', 14, 'bold')).pack(pady=(0, 10))
        self.webcam_display = tk.Label(left_frame, bg='black')
        self.webcam_display.pack(fill=tk.BOTH, expand=True)
        self.emotion_status = ttk.Label(left_frame, text="Detecting Emotion...", font=('Calibri', 12), foreground='blue')
        self.emotion_status.pack(pady=10)

        # Right side - Emotion Image
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        ttk.Label(right_frame, text="Your Emotion's Avatar", font=('Calibri', 14, 'bold')).pack(pady=(0, 10))
        self.emotion_display = tk.Label(right_frame, bg='lightgray')
        self.emotion_display.pack(fill=tk.BOTH, expand=True)
        instructions = ttk.Label(right_frame, text="Instructions:\n1. Look at the webcam\n2. Your emotion will be detected\n3. The avatar will react to your feelings", font=('Calibri', 10), justify=tk.LEFT)
        instructions.pack(pady=20)

    def update_webcam(self):
        """
        This function runs on the MAIN GUI thread.
        Its only job is to read frames from the webcam and display the
        latest processed frame from the background thread.
        """
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Share the latest raw frame with the processing thread
            with self.frame_lock:
                self.latest_frame = frame.copy()

            # Get the latest processed frame to display
            display_frame = None
            with self.frame_lock:
                if self.processed_frame is not None:
                    display_frame = self.processed_frame
                else:
                    display_frame = frame  # Fallback to raw frame if not processed yet

            # Convert for Tkinter display
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1000, 750))
            photo = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
            self.webcam_display.configure(image=photo)
            self.webcam_display.image = photo

        # Schedule the next update
        self.root.after(16, self.update_webcam)  # Aims for ~60 FPS

    def video_processing_loop(self):
        """
        This function runs on a BACKGROUND thread.
        It handles all heavy processing to prevent freezing the GUI.
        """
        emotion_detection_interval = 3.0  # Seconds between emotion analysis
        last_emotion_time = 0

        while self.running:
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            # --- 1. Face Detection (runs continuously) ---
            faces, frame_with_boxes = self.detect_faces(frame_to_process)

            # --- 2. Emotion Detection (throttled for performance) ---
            current_time = time.time()
            if faces and (current_time - last_emotion_time > emotion_detection_interval):
                last_emotion_time = current_time
                # Use the largest face for emotion detection
                largest_face = faces[0]
                x1, y1, x2, y2 = largest_face['coords']
                face_region = frame_to_process[y1:y2, x1:x2]

                if face_region.size > 0:
                    emotions = self.enhanced_emotion_detection(face_region)
                    if emotions:
                        detected_emotion = max(emotions, key=emotions.get)
                        if detected_emotion != self.current_emotion:
                            self.current_emotion = detected_emotion
                            # Schedule GUI update on the main thread
                            self.root.after(0, self.update_emotion_display)
                        
                        # Convert numpy float32 to regular float for compatibility
                        converted_emotions = {}
                        for emotion, confidence in emotions.items():
                            converted_emotions[emotion] = float(confidence)
                        
                        self.last_stable_emotion_data = sorted(converted_emotions.items(), key=lambda item: item[1], reverse=True)
                        self.last_emotion_update_time = current_time
                        
                        # Debug: Print emotion data
                        print(f"Emotion data: {self.last_stable_emotion_data}")

                        status_text = f"Primary Emotion: {detected_emotion.title()}"
                        self.root.after(0, lambda s=status_text: self.emotion_status.configure(text=s, foreground='green'))

            # --- 3. Draw Emotion Overlay (throttled for performance) ---
            # Only draw overlay when we have fresh data or every few frames
            if (self.last_stable_emotion_data and len(self.last_stable_emotion_data) > 0 and
                time.time() - self.last_emotion_update_time < (emotion_detection_interval + 1.0)):
                
                # Only draw overlay every 10th frame to reduce CPU usage
                if not hasattr(self, 'overlay_frame_count'):
                    self.overlay_frame_count = 0
                
                self.overlay_frame_count += 1
                if self.overlay_frame_count >= 10:  # Draw every 10th frame
                    frame_with_boxes = self.add_emotion_overlay(frame_with_boxes, self.last_stable_emotion_data)
                    self.overlay_frame_count = 0

            # --- 4. Share the fully processed frame back to the main thread ---
            with self.frame_lock:
                self.processed_frame = frame_with_boxes
            
            time.sleep(0.01) # Yield the CPU briefly

    def detect_faces(self, frame):
        """Detects faces using YOLO, draws bounding boxes, and returns face data."""
        if self.yolo_model is None:
            return [], frame

        faces = []
        frame_with_boxes = frame.copy()
        try:
            results = self.yolo_model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        if confidence > 0.5:
                            area = (x2 - x1) * (y2 - y1)
                            faces.append({'coords': (x1, y1, x2, y2), 'area': area})
            
            if faces:
                faces.sort(key=lambda x: x['area'], reverse=True)
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face['coords']
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)
                    thickness = 4 if i == 0 else 2
                    label = "TARGET" if i == 0 else "Face"
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            print(f"Face detection error: {e}")
        return faces, frame_with_boxes

    def add_emotion_overlay(self, frame, emotion_data):
        """Draws the semi-transparent emotion analysis overlay on the frame."""
        try:
            # Debug: Print what we're trying to draw (reduced frequency)
            if not hasattr(self, 'debug_count'):
                self.debug_count = 0
            self.debug_count += 1
            if self.debug_count % 50 == 0:  # Print every 50th call
                print(f"Drawing overlay with data: {emotion_data[:3] if emotion_data else 'None'}")
            
            h, w, _ = frame.shape
            overlay = frame.copy()
            start_x, start_y = w - 400, 10
            
            # Draw semi-transparent background
            cv2.rectangle(overlay, (start_x, start_y), (w - 10, start_y + 250), (0, 0, 0), -1)
            cv2.rectangle(overlay, (start_x, start_y), (w - 10, start_y + 250), (255, 255, 255), 2)
            
            # Add title
            cv2.putText(overlay, "WHAT YOU'RE FEELING", (start_x + 15, start_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            y_offset = 60
            for emotion, confidence in emotion_data[:5]: # Display top 5 emotions
                if confidence < 1: continue # Skip very low confidence scores
                text = f"{emotion.title()}: {confidence:.1f}%"
                bar_width = int((confidence / 100) * 250)
                color = (0, 255, 0) if confidence > 50 else (0, 255, 255)
                
                # Draw emotion bar
                cv2.rectangle(overlay, (start_x + 15, start_y + y_offset - 8),
                              (start_x + 15 + bar_width, start_y + y_offset + 8), color, -1)
                
                # Draw text with background
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_bg_x = start_x + 15 + bar_width + 5
                cv2.rectangle(overlay, (text_bg_x, start_y + y_offset - 8),
                              (text_bg_x + text_size[0] + 10, start_y + y_offset + 8), (0, 0, 0), -1)
                cv2.putText(overlay, text, (text_bg_x + 5, start_y + y_offset + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 35
            
            # Blend overlay with original frame
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            return frame  # Return the modified frame
        except Exception as e:
            print(f"Emotion overlay error: {e}")
            return frame  # Return original frame if error

    def enhanced_emotion_detection(self, face_region):
        """Analyzes a face region for emotions using FER or DeepFace."""
        try:
            # Try FER first for its sensitivity
            if self.fer_detector:
                result = self.fer_detector.detect_emotions(face_region)
                if result:
                    return result[0]['emotions']
            
            # Fallback to DeepFace
            analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            if analysis:
                return analysis[0]['emotion']
        except Exception as e:
            print(f"Enhanced emotion detection error: {e}")
        return None

    def update_emotion_display(self):
        """Updates the emotion image on the right side of the GUI."""
        image_path = self.emotion_images.get(self.current_emotion, 'emotion_images/neutral.png')
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).resize((1400, 1050), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.emotion_display.configure(image=photo)
                self.emotion_display.image = photo
            except Exception as e:
                print(f"Error loading emotion image: {e}")

    def on_closing(self):
        """Handles application closing cleanly."""
        print("Closing application...")
        self.running = False
        time.sleep(0.5) # Give the thread a moment to close
        self.cap.release()
        self.root.destroy()

    def run(self):
        """Starts the Tkinter application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    os.makedirs('emotion_images', exist_ok=True)
    print("Starting Emotion Detection App...")
    app = EmotionDetector()
    app.run()
