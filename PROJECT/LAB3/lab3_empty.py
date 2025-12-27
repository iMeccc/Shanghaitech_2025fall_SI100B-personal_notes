import numpy as np
import cv2

def conv2d(x: np.ndarray, kernel: np.ndarray, stride=1, padding=0) -> np.ndarray:
    H, W = x.shape
    K = kernel.shape[0]
    
    # Compute output size
    out_H = (H - K) // stride + 1
    out_W = (W - K) // stride + 1
    
    # Initialize output
    out = np.zeros((out_H, out_W))
    
    # Slide the kernel
    for i in range(out_H):
        for j in range(out_W):
            region = x[i*stride : i*stride + K, j*stride : j*stride + K]
            out[i, j] = np.sum(region * kernel)
    
    return out

def sigmod(X) -> np.ndarray:
    x = 1.0/(1 + np.exp(-X))
    return x

def max_pool_2d(x, pool_size=2, stride=2):
    H, W = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    out = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            region = x[i*stride : i*stride + pool_size, j*stride : j*stride + pool_size]
            out[i, j] = np.max(region)
    
    return out

def flatten(X) -> np.ndarray: 
    return X.flatten()

def full_connect(X, W, b) -> np.ndarray: 
    out = np.dot(X, W) + b
    return out

if __name__ == '__main__':
  #Read test case
  img = cv2.imread(r"PROJECT\Lab3\sad.jpg", cv2.IMREAD_GRAYSCALE) 
  #Normailize to [-1, 1)
  assert img is not None, "Image not found"
  img = img / 127.0 - 1.0
  #Set parameters
  con_kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
  W = np.array([[0.01] * 529, [0.02] * 529, [0.03] * 529]).T 
  b = np.array([1,1,1])

  # 1. Convolution
  out = conv2d(img, con_kernel)
  print(f"After Conv2D: {out.shape}")

  # 2. Activation (sigmod)
  out = sigmod(out)
  print(f"After sigmod: {out.shape}")

  # 3. Max Pooling
  out = max_pool_2d(out, pool_size=2, stride=2)
  print(f"After MaxPool2D: {out.shape}")

  # 4. Flatten
  out = flatten(out)
  print(f"After Flatten: {out.shape}")

  # 5. Fully Connected
  prob = full_connect(out, W, b)
  print(f"Final Output Scores: {prob}")