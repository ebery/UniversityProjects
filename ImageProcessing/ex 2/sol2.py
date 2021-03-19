import numpy as np
import imageio as io
import skimage.color as sk
import scipy.signal as sc

BITS = 256
INVERSE = True
DY_DERIVE = np.array([[1], [0], [-1]])
DX_DERIVE = DY_DERIVE.T
GAUSSIAN_KERNEL = np.array([[1, 1]])


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation: RGB or grayscale.
    :param filename: A path to an image that we want to work on, string
    :param representation: Tells in which representation the output should be: 1 for gray-scale, 2 for RGB.
    :return: Image in given representation
    """
    image = io.imread(filename)
    is_gray = True  # indicates if the image is grayscale or not
    image_float = (image / (BITS - 1)).astype(np.float64)
    if len(image.shape) > 2 and image.shape[-1] == 3:  # image is RGB
        is_gray = False
    if (not is_gray) and (representation == 1):
        return sk.rgb2gray(image_float)
    elif (is_gray and (representation == 1)) or representation == 2:
        return image_float


def DFT_matrix(N, inverse=False):
    """
    Creates the DFT matrix
    :param N: The size of the matrix needed (the size of the signal)
    :param inverse: True if IDFT Matrix is wanted, default is False.
    :return: an array of dtype complex128 with shape (N,N)
    """
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2 * np.pi * 1J / N)
    if inverse:
        omega = np.power(omega, -1)
    W = np.power(omega, i * j).astype(np.complex128)
    return W


def DFT(signal):
    """
    Transforms a 1D discrete signal to its Fourier representation.
    :param signal: n array of dtype float64 with shape (N,1).
    :return: Fourier transform of a signal - an array of dtype complex128 with shape (N,1).
    """
    N = np.size(signal, 0)
    M = DFT_matrix(N)
    return np.dot(M, signal).astype(np.complex128)


def IDFT(fourier_signal):
    """
    Transforms a 1D discrete signal to its inverse Fourier representation.
    :param fourier_signal: n array of dtype float64 with shape (N,1).
    :return: Inverse fourier transform of a signal.
    """
    N = np.size(fourier_signal, 0)
    M = DFT_matrix(N, INVERSE)
    f = np.dot(((1/N)*M), fourier_signal)
    return f


def DFT2(image):
    """
    Converts a 2D discrete signal with shape (M, N) to its Fourier representation.
    :param image: Grays-cale image of dtype float64 of shape (M, N).
    :return: Fourier transform of the image - 2D array of dtype complex128.
    """
    N = np.size(image, 1)
    M = np.size(image, 0)
    row_MAT = DFT_matrix(N)
    col_MAT = DFT_matrix(M)
    return np.dot(col_MAT, np.dot(image, row_MAT)).astype(np.complex128)


def IDFT2(fourier_image):
    """
    Converts a 2D discrete signal to its inverse Fourier representation.
    :param fourier_image: An image converted to its fourier representation.
    :return: Inverse Fourier transform of an image
    """
    N = np.size(fourier_image, 1)
    M = np.size(fourier_image, 0)
    row_MAT = DFT_matrix(N, INVERSE)
    col_MAT = DFT_matrix(M, INVERSE)
    return 1/(N*M) * np.dot(col_MAT, np.dot(fourier_image, row_MAT))


def conv_der(im):
    """
    Computes the magnitude of image derivatives
    :param im: A gray-scale image of type float64
    :return:
    """
    dx = sc.convolve2d(im, DX_DERIVE, mode='same')
    dy = sc.convolve2d(im, DY_DERIVE, mode='same')
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def fourier_der(im):
    """
    Computes the magnitude of image derivatives using Fourier transform.
    :param im: A gray-scale image of type float64
    :return: The magnitude of the derivative, with the same dtype and shape
    """
    M = np.size(im, 0)
    N = np.size(im, 1)
    u, v = np.meshgrid(np.arange(-(N//2), np.ceil(N/2)), np.arange(-(M//2), np.ceil(M/2)))
    dx = ((2*np.pi*1j)/N) * IDFT2(np.fft.ifftshift(u*np.fft.fftshift(DFT2(im))))
    dy = ((2*np.pi*1j)/N) * IDFT2(np.fft.ifftshift(v*np.fft.fftshift(DFT2(im))))
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def get_gaussian_kernel(kernel_size):
    """
    The function calculates a gaussian kernel in given size.
    :param kernel_size: Size of the wanted kernel.
    :return: Gaussian kernel in space domain in wanted size.
    """
    kernel = [[1]]
    for i in range(kernel_size - 1):
        kernel = sc.convolve2d(kernel, GAUSSIAN_KERNEL)
    kernel = np.asarray(kernel)
    return sc.convolve2d(kernel, kernel.T) / np.sum(sc.convolve2d(kernel, kernel.T))


def blur_spatial(im, kernel_size):
    """
    The function performs image blurring using 2D convolution between the image f and a gaussian
    kernel g.
    :param im: The input image to be blurred (grayscale float64 image).
    :param kernel_size: Size of the gaussian kernel in each dimension (an odd integer).
    :return: Blurred image.
    """
    if kernel_size == 1:
        return im
    kernel_mat = get_gaussian_kernel(kernel_size)
    return sc.convolve2d(im, kernel_mat, mode='same', boundary='symm')


def blur_fourier(im, kernel_size):
    """
    Write a function that performs image blurring with gaussian kernel in Fourier space.
    :param im: The input image to be blurred (grayscale float64 image).
    :param kernel_size: - Size of the gaussian kernel in each dimension (an odd integer).
    :return:
    """
    if kernel_size == 1:
        return im
    kernel = get_gaussian_kernel(kernel_size)
    m, n = im.shape
    im_size_kernel = np.zeros((m, n), np.float64)
    im_size_kernel[
        int(m//2 - kernel_size//2):int(m//2 + kernel_size//2 + 1),
        int(n//2 - kernel_size//2):int(n//2 + kernel_size//2 + 1)
    ] = kernel
    fourier_kernel = DFT2(im_size_kernel)
    fourier_im = DFT2(im)
    return np.fft.ifftshift(IDFT2(np.multiply(fourier_kernel, fourier_im))).real
