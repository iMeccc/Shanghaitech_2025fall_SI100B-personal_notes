import cv2

# Load pre-trained cascade file
face_cascade = cv2.CascadeClassifier(r'/Users/meccc/Repository/Shanghaitech/2025fall_SI100B/PROJECT/LAB2/haar-cascade-files/haarcascade_frontalface_default.xml')
# Read image and convert to grayscale
image = cv2.imread(r'/Users/meccc/Repository/Shanghaitech/2025fall_SI100B/PROJECT/LAB2/demo.png')
assert image is not None
# Convert to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)