import numpy as np
import cv2
import os

def conv4d(input_image,  kernel, stride=1, padding=0):
    batch_size, H, W, cin = input_image.shape
    cout, K_H, K_W, cin_k = kernel.shape
    assert cin == cin_k, "kernel channel should match input channel"

    H_out = (H - K_H)//stride +1    
    W_out = (W - K_W)//stride +1
    out = np.zeros((batch_size, H_out, W_out, cout))
    for b in range(batch_size):
        for c_out in range(cout):
            for i in range(H_out):
                for j in range(W_out):
                    region = input_image[b, i*stride:i*stride+K_H, j*stride:j*stride+K_W,:]
                    out[b,i,j,c_out] = np.sum(region*kernel[c_out,:,:,:])
    return out

def MaxPool(x, size):
    batch_size, H, W, c = x.shape
    stride = size
    H_out = H//stride
    W_out = W//stride
    out = np.zeros((batch_size, H_out, W_out, c))
    for b in range(batch_size):
        for ch in range(c):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    region = x[b, h_start : h_start + size, w_start : w_start + size, ch]
                    out[b, i, j, ch] = np.max(region)
    return out

def ReLU(x):
    return np.maximum(0, x)

def flatten(x):
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)

def full_connect(x, weights, biases):
    output = x.dot(weights) + biases  
    return output

if __name__ == '__main__':
    batch_size = 4

    # 1 read image 
    # (Batch_size, Height, Width, Channels)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = np.array([cv2.imread(os.path.join(script_dir,'sad.jpg')),
                        cv2.imread(os.path.join(script_dir,'sad1.jpg')),
                        cv2.imread(os.path.join(script_dir,'sad2.jpg')),
                        cv2.imread(os.path.join(script_dir,'sad3.jpg'))])
    print(input_image.shape)

    # 2 load weights from local 
    conv_w = np.load(os.path.join(script_dir,"conv_w.npy"))
    fc_w = np.load(os.path.join(script_dir,"fc.npy"))

    # Using maxpool as pooling fucntion and using ReLU as activation function.
    # 3 inference
    convolution_img = conv4d(input_image, conv_w, stride=1, padding=0)
    relu_img = ReLU(convolution_img)
    pooled_img = MaxPool(relu_img, size=2)
    flatten_img = flatten(pooled_img)

    num_classes = fc_w.shape[1]
    biases = np.zeros((num_classes,))
    output = full_connect(flatten_img, fc_w, biases)

    print(output)