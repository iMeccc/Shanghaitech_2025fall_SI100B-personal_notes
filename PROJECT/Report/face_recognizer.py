import cv2
import os
import numpy as np

class FaceRecognizer:
    """
    A class to handle face recognition using OpenCV's LBPH recognizer.
    It encapsulates the face detection, training, and recognition logic.
    """
    def __init__(self, cascade_path=None):
        # Load the Haar Cascade for face detection once during initialization.
        if cascade_path is None:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar Cascade file not found at: {cascade_path}")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.target_size = (100, 100) # Standard size for faces

    def _preprocess_image(self, image):
        """
        Detects a single face in an image and returns the preprocessed ROI.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None, None # Return None if no face is detected
        
        # Assume the largest detected face is the one of interest
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_roi = gray[y:y+h, x:x+w]
        
        return cv2.resize(face_roi, self.target_size), (x, y, w, h)

    def train(self, train_dir):
        """
        Trains the LBPH recognizer on a directory of images of a single person.
        """
        print("Starting training process...")
        faces, labels = [], []
        target_label = 0 # This system is designed for one target person

        image_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            raise ValueError(f"No valid images found in the training directory: {train_dir}")

        for filename in image_files:
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
            
            face_roi, _ = self._preprocess_image(img)
            if face_roi is not None:
                faces.append(face_roi)
                labels.append(target_label)
                print(f"Processed face from {filename}")
        
        if not faces:
            raise ValueError("No faces were detected in the training images.")

        self.recognizer.train(faces, np.array(labels))
        print("Model training complete!")

    def recognize(self, image, threshold=80):
        """
        Recognizes faces in a given image and returns predictions.
        """
        face_roi, face_pos = self._preprocess_image(image)
        
        if face_roi is None:
            print("No face detected in the test image.")
            return None, None

        predicted_label, confidence = self.recognizer.predict(face_roi)
        print(f"Confidence: {confidence:.2f} (lower is better)")
        
        is_target = predicted_label == 0 and confidence < threshold
        
        result = {
            "is_target": is_target,
            "position": face_pos,
            "confidence": confidence
        }
        return result
    
    def save_model(self, path="recognizer.yml"):
        self.recognizer.save(path)
        print(f"Recognizer model saved to {path}")

    def load_model(self, path="recognizer.yml"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Recognizer model file not found at: {path}")
        self.recognizer.read(path)
        print(f"Recognizer model loaded from {path}")


def draw_result(image, result):
    """
    Draws the recognition result on the image.
    """
    if result is None:
        return image
        
    (x, y, w, h) = result["position"]
    is_target = result["is_target"]
    
    color = (0, 255, 0) if is_target else (0, 0, 255)
    label = "Target" if is_target else "Unknown"
    
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image


if __name__ == "__main__":
    # --- Hardcoded configuration (edit these paths as needed) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CASCADE_PATH = r"C:\CODE\ShanghaiTech\2025fall_SI100B\PROJECT\LAB2\haar-cascade-files\haarcascade_frontalface_default.xml"
    TRAIN_DIR = r"C:\CODE\ShanghaiTech\2025fall_SI100B\PROJECT\Report\train"  # set to None to skip training
    TEST_IMG = r"C:\CODE\ShanghaiTech\2025fall_SI100B\PROJECT\Report\test\1.png"
    SAVE_MODEL = os.path.join(BASE_DIR, 'recognizer.yml')
    LOAD_MODEL = None  # e.g. os.path.join(BASE_DIR, 'recognizer.yml') to load instead of training
    THRESHOLD = 80

    try:
        recognizer = FaceRecognizer(cascade_path=CASCADE_PATH)

        # Decide whether to load a pre-trained model or train a new one
        if LOAD_MODEL:
            recognizer.load_model(LOAD_MODEL)
        elif TRAIN_DIR:
            recognizer.train(TRAIN_DIR)
            recognizer.save_model(SAVE_MODEL)
        else:
            raise ValueError("Set TRAIN_DIR to train a model or set LOAD_MODEL to load an existing one.")

        # Perform recognition
        print("\n===== Starting Recognition =====")
        test_image = cv2.imread(TEST_IMG)
        if test_image is None:
            raise FileNotFoundError(f"Could not read test image: {TEST_IMG}")

        recognition_result = recognizer.recognize(test_image, THRESHOLD)

        if recognition_result:
            result_image = draw_result(test_image, recognition_result)

            # Display and save the final image
            cv2.imshow("Recognition Result", result_image)
            cv2.waitKey(0)
            cv2.imwrite("recognition_result.jpg", result_image)
            print("Result image saved to recognition_result.jpg")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()