import numpy as np
from scipy.signal import fftconvolve

def oneHot(indecies:list[int], depth:int):
    oneHot_arr = np.zeros((len(indecies), depth))
    for first, second in enumerate(indecies):
        oneHot_arr[first][second] = 1
    return oneHot_arr

def argmax(arr:list[float]):
    rec = 0
    index = 0
    for i in arr:
        if i > rec:
            rec = i
            index = np.where(arr == i)[0][0]
    return index

# image processing ----------------------------------------

def im2col(images, kernel_dim, stride=1, padding=0):
    #kernel_dim is kh * kw
    patch_size = kernel_dim * kernel_dim * images.shape[1]
    output_height = (images.shape[2] - kernel_dim + 2 * padding) // stride + 1
    output_width = (images.shape[3] - kernel_dim + 2 * padding) // stride + 1
    output_size = output_width * output_height

    im2col = np.zeros((images.shape[0], output_size, patch_size))
    images = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    index = 0
    for i in range(output_height):
        for j in range(output_width):
            I, J = i * stride, j * stride
            im2col[:,index:index+1] = images[:, :, I:I + kernel_dim, J:J + kernel_dim].reshape((images.shape[0], 1, patch_size))
            index += 1
    return im2col, (output_height, output_width)

def fftConvolve(images, kernels):
    output_shape = (images.shape[2] - kernels.shape[2] + 1, images.shape[3] - kernels.shape[3] + 1)
    conv_images = np.zeros((images.shape[0], kernels.shape[0], output_shape[0], output_shape[1]))
    for i, image in enumerate(images):
        for j, kernel in enumerate(kernels):
            conv_images[i][j] = fftconvolve(image, kernel, mode='valid')
    return conv_images

def convolve(img, kernels, stride=1, padding=0, bias=None):
    kernel_dim = kernels.shape[2]
    kernel_matrix = kernels.reshape(kernels.shape[0], kernels.shape[1] * kernels.shape[2] * kernels.shape[3])
    img_matrix, output_shape = im2col(img, kernel_dim, stride=stride, padding=padding)

    output = img_matrix @ kernel_matrix.T
    output = output.transpose(0, 2, 1)
    if isinstance(bias, np.ndarray):
        output = output + bias[np.newaxis, :, np.newaxis]
    output = output.reshape((output.shape[0], output.shape[1], output_shape[0], output_shape[1]))
    return output

def up_sample(x, stride):
    N, C, H, W = x.shape
    H_up = (H - 1) * stride + 1
    W_up = (W - 1) * stride + 1

    up_sample = np.zeros((N, C, H_up, W_up), dtype=x.dtype)

    up_sample[:, :, ::stride, ::stride] = x
    return up_sample