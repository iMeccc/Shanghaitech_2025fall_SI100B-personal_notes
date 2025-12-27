import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torchvision import transforms

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
# --- 1. 模型定义 ---
from LAB4.LAB4_empty.my_net.classify import emotionNet

# --- 2. 核心逻辑类 ---
class EmotionDetector:
    def __init__(self, cascade_path, model_path, device='cpu'):
        self.classes = ['happy', 'neutral', 'sad']
        self.device = torch.device(device)
        self._load_models(cascade_path, model_path)
        self._setup_transform()

    def _load_models(self, cascade_path, model_path):
        """私有方法，用于加载所有模型。"""
        print(f"Loading Haar Cascade from: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Cannot load Haar Cascade model from {cascade_path}")

        print(f"Loading emotion model from: {model_path}")
        self.model = emotionNet(False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Models loaded successfully.")

    def _setup_transform(self):
        """私有方法，用于设置图像预处理流程。"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_predict(self, image):
        """
        处理单张图像，检测人脸并标注情绪。
        """
        display_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            return display_image, []

        face_rois, face_coords = self._extract_faces(image, faces)
        input_batch = self._preprocess_faces(face_rois)
        predictions = self._predict_batch(input_batch)
        annotated_image = self._draw_annotations(display_image, face_coords, predictions)
        
        return annotated_image, predictions

    def _extract_faces(self, image, faces):
        face_rois = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
        face_coords = list(faces)
        return face_rois, face_coords

    def _preprocess_faces(self, face_rois):
        batch_tensors = [self.transform(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) for roi in face_rois]
        return torch.stack(batch_tensors).to(self.device)

    def _predict_batch(self, input_batch):
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, indices = torch.max(probabilities, 1)
        
        predictions = []
        for i in range(len(indices)):
            predictions.append({
                "label": self.classes[int(indices[i].item())],
                "confidence": confidences[i].item()
            })
        return predictions

    def _draw_annotations(self, image, coords, predictions):
        for i, (x, y, w, h) in enumerate(coords):
            pred = predictions[i]
            label_text = f"{pred['label']}: {pred['confidence']*100:.1f}%"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return image


# --- 4. 主程序入口 ---
def main():
    # --- 使用基于脚本位置的相对路径 ---
    # 1. 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 构建到各个文件的路径
    # 假设所有文件都与此脚本在同一目录下
    cascade_file = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
    model_file = os.path.join(script_dir, "face_expression_excel.pth") 
    image_file = os.path.join(script_dir, "demo.png") 
    output_file = os.path.join(script_dir, "result.jpg")

    # ------------------------------------

    # 初始化检测器
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # 检查文件是否存在
        for f in [cascade_file, model_file, image_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        detector = EmotionDetector(cascade_path=cascade_file, model_path=model_file, device=device)
        
        image = cv2.imread(image_file)
        if image is None:
            raise IOError(f"Could not read the image file: {image_file}")

        result_image, predictions = detector.detect_and_predict(image)
        
        print(f"Processing complete. Found {len(predictions)} faces.")
        for i, pred in enumerate(predictions):
            print(f"  Face {i+1}: {pred['label']} ({pred['confidence']*100:.1f}%)")
            
        cv2.imwrite(output_file, result_image)
        print(f"Result saved to {output_file}")

        cv2.imshow('Emotion Detection Result', result_image)
        print("Press any key to exit...")
        cv2.waitKey(0)

    except (IOError, FileNotFoundError, RuntimeError) as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()