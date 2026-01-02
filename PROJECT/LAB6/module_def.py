import cv2
import torch
from torchvision import transforms
import torch.nn as nn

class emotionNet(nn.Module):
    def __init__(self, printtoggle):
        super().__init__()
        self.print = printtoggle

        # step1:
        # Define the functions you need: convolution, pooling, activation, and fully connected functions.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # 根据图示
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0) # 根据图示
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.LeakyReLU()

        self.fc1 = nn.Linear(in_features=4096, out_features=3) 
        self.relu4 = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)


    def forward(self, x):
        #Step 2
        # Using the functions your defined for forward propagate
        # First block
        # convolution -> maxpool -> relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Second block
        # convolution -> maxpool -> relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Third block
        # convolution -> maxpool -> relu
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        # Flatten for linear layers
        x = torch.flatten(x, start_dim=1)

        # fully connect layer
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        
        return x

class EmotionDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmotionDetector, cls).__new__(cls)
        else:
            print("Returning the existing EmotionDetector instance...")
        return cls._instance
    
    def __init__(self, cascade_path, model_path, device='cpu'):
        if hasattr(self, 'model'):
            return # Avoid re-initialization
        self.classes = ['happy', 'neutral', 'sad']
        self.device = torch.device(device)
        self._load_models(cascade_path, model_path)
        self._setup_transform()

    def _load_models(self, cascade_path, model_path):
        # load cascade model and emotion model
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
        # set up image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_predict(self, image):
        # process a single image, detect faces and annotate emotions
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
        # extract face ROIs and their coordinates
        face_rois = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
        face_coords = list(faces)
        return face_rois, face_coords

    def _preprocess_faces(self, face_rois):
        # preprocess face ROIs into a batch tensor
        batch_tensors = [self.transform(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) for roi in face_rois]
        return torch.stack(batch_tensors).to(self.device)

    def _predict_batch(self, input_batch):
        # predict emotions for a batch of face ROIs
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
        # draw rectangles and labels on the image
        for i, (x, y, w, h) in enumerate(coords):
            pred = predictions[i]
            label_text = f"{pred['label']}: {pred['confidence']*100:.1f}%"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return image