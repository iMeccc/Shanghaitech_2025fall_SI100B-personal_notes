import cv2
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Generator an empty image with 480 height and 640 width
    ###FILL red when x in range [0, 160] and y in range [0, 120]
    ###FILL GREEN when x in range [160, 320] and y in range [120, 240]
    ###FILL GREEN when x in range [320, 480] and y in range [240, 360]
    ###FILL GRAY[128, 128, 128] when x in range [480, 640] and y in range [360, 480]
    blanks = np.zeros((480, 640, 3), dtype=np.uint8)
    blanks[0:120, 0:160] = (0, 0, 255)
    blanks[120:240, 160:320] = (0, 255, 0)
    blanks[240:360, 320:480] = (0, 255, 0)
    blanks[360:480, 480:640] = (128, 128, 128)

    cv2.imwrite("create.png", blanks)
    cv2.imshow("src", blanks)
    cv2.waitKey(0)