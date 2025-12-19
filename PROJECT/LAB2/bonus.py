import cv2
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haar-cascade-files', 'haarcascade_lefteye_2splits.xml')
img_path = os.path.join(script_dir, 'demo.png')

eye_cascade = cv2.CascadeClassifier(cascade_path)
img = cv2.imread(img_path)
if img is None:
	raise FileNotFoundError(f'Image not found at path: {img_path}')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes, numDetections = eye_cascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

assert isinstance(eyes, np.ndarray)
assert isinstance(numDetections, np.ndarray)
scores_list = numDetections.flatten().tolist()
boxes_list = eyes.tolist()
nms_threshold = 0.3
score_threshold = 0.0
indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, score_threshold, nms_threshold)
assert isinstance(indices, np.ndarray)

if len(indices) > 0:
    for i in indices.flatten():
        (x, y, w, h) = eyes[i]
        center = (x + w // 2, y + h // 2)
        radius = int(round((w + h) * 0.25))
        cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.imshow('Detected Eyes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


