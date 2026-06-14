import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from typing import Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime


class EmotionDetector:
    def __init__(self, model_path: str = "saved_model.keras"):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = ["Angry", "Happy", "Neutral", "Sad"]
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Detect input shape from model
        input_shape = self.model.input_shape
        self.input_size = input_shape[1]  # Height/Width (assumes square)
        self.channels = input_shape[-1] if len(input_shape) > 3 else 1
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model input shape: {input_shape}")
        print(f"✓ Emotions: {', '.join(self.emotions)}")

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        face_resized = cv2.resize(face, (self.input_size, self.input_size))
        
        # Handle channels based on model requirement
        if self.channels == 1:
            # Convert to grayscale for single channel
            if len(face_resized.shape) == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        elif self.channels == 3:
            # Ensure BGR for 3 channels
            if len(face_resized.shape) == 2:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
        
        # Normalize pixel values to 0-1
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions as needed
        if len(face_normalized.shape) == 2:
            face_normalized = np.expand_dims(face_normalized, -1)
        
        face_input = np.expand_dims(face_normalized, 0)
        
        return face_input

    def predict_emotion(self, face: np.ndarray) -> Tuple[str, float]:
        """Predict emotion from face image."""
        try:
            face_input = self.preprocess_face(face)
            predictions = self.model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx]) * 100
            emotion = self.emotions[emotion_idx]
            return emotion, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0

    def process_image(self, image_path: str) -> None:
        """Detect emotions in a single image."""
        print(f"\n📷 Processing image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("⚠️  No faces detected in image")
            return
        
        print(f"✓ Found {len(faces)} face(s)")
        
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            emotion, confidence = self.predict_emotion(face)
            
            # Draw rectangle and label
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"  Face {i+1}: {label}")
        
        # Save result
        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, image)
        print(f"✓ Result saved to: {output_path}")
        
        # Display result
        self._display_image(image, f"Emotion Detection - {Path(image_path).name}")

    def webcam_detection(self, duration: int = 30) -> None:
        """Real-time emotion detection from webcam."""
        print(f"\n📹 Starting webcam detection ({duration}s)...")
        print("Press 'q' to exit early")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        start_time = datetime.now()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion, confidence = self.predict_emotion(face)
                
                # Draw rectangle and label
                color = self._get_emotion_color(emotion)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{emotion} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add timestamp and frame count
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Emotion Detector - Webcam", frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if (datetime.now() - start_time).total_seconds() > duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✓ Webcam detection finished ({frame_count} frames processed)")

    def batch_process(self, directory: str) -> None:
        """Process all images in a directory."""
        print(f"\n📁 Batch processing images from: {directory}")
        
        image_dir = Path(directory)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {directory}")
            return
        
        print(f"✓ Found {len(image_files)} image(s)")
        
        results = []
        for image_path in sorted(image_files):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = image[y:y+h, x:x+w]
                    emotion, confidence = self.predict_emotion(face)
                    results.append({
                        'file': image_path.name,
                        'emotion': emotion,
                        'confidence': confidence
                    })
                    print(f"  {image_path.name}: {emotion} ({confidence:.1f}%)")
        
        # Summary statistics
        if results:
            print("\n📊 Summary:")
            emotion_counts = {}
            for r in results:
                emotion = r['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            for emotion, count in sorted(emotion_counts.items()):
                print(f"  {emotion}: {count}")

    @staticmethod
    def _get_emotion_color(emotion: str) -> Tuple[int, int, int]:
        """Get BGR color for emotion."""
        colors = {
            "Angry": (0, 0, 255),      # Red
            "Happy": (0, 255, 0),      # Green
            "Neutral": (255, 255, 0),  # Cyan
            "Sad": (255, 0, 0)         # Blue
        }
        return colors.get(emotion, (255, 255, 255))

    @staticmethod
    def _display_image(image: np.ndarray, title: str = "Image") -> None:
        """Display image using matplotlib."""
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition")
    parser.add_argument("--mode", choices=["image", "webcam", "batch"],
                       default="webcam", help="Detection mode")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--batch", type=str, help="Path to image directory")
    parser.add_argument("--duration", type=int, default=30,
                       help="Webcam duration in seconds")
    parser.add_argument("--model", type=str, default="saved_model.keras",
                       help="Path to model file")
    
    args = parser.parse_args()
    
    try:
        detector = EmotionDetector(args.model)
        
        if args.mode == "image":
            if not args.image:
                print("❌ Please provide image path with --image")
                return
            detector.process_image(args.image)
        
        elif args.mode == "webcam":
            detector.webcam_detection(args.duration)
        
        elif args.mode == "batch":
            if not args.batch:
                print("❌ Please provide directory path with --batch")
                return
            detector.batch_process(args.batch)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
