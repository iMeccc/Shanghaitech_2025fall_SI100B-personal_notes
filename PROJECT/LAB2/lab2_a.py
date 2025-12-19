import cv2
image = cv2.imread(r'C:\CODE\ShanghaiTech\2025fall_SI100B\PROJECT\LAB1\create.png')
assert image is not None

print(image.shape)
cv2.rectangle(image,(80,60),(560,420),(0,255,0),3)
cropped_image = image[30:450, 40:600]

height, width = cropped_image.shape[:2]
new_width = width // 2
new_height = height // 2
resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

text = "I love SI100B"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
color = (255, 255, 255)
thickness = 2
# 计算文本大小
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

# 计算文本位置（居中）
text_x = (resized_image.shape[1] - text_width) // 2
text_y = (resized_image.shape[0] + text_height) // 2

# 在图像上绘制文本
cv2.putText(resized_image, text, (text_x, text_y), font, font_scale, color, thickness)

cv2.imshow('result',resized_image)
cv2.waitKey(0)