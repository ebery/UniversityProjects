import numpy as np
import imageio
import skimage.color as sk
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import scipy.signal as sc


BITS = 256
GAUSSIAN_KERNEL = np.array([[1, 1]])
MIN_SHAPE_SIZE = 16


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation: RGB or grayscale.
    :param filename: A path to an image that we want to work on, string
    :param representation: Tells in which representation the output should be: 1 for gray-scale, 2 for RGB.
    :return: Image in given representation
    """
    image = imageio.imread(filename)
    is_gray = True  # indicates if the image is grayscale or not
    image_float = (image / (BITS - 1)).astype(np.float64)
    if len(image.shape) > 2 and image.shape[-1] == 3:  # image is RGB
        is_gray = False
    if (not is_gray) and (representation == 1):
        return sk.rgb2gray(image_float)
    elif (is_gray and (representation == 1)) or representation == 2:
        return image_float


def get_gaus_filter(filter_size):
    """
    Calculates a gaussian filter in wanted size.
    :param filter_size: The wanted size of the filter.
    :return: Row vector of shape (1, filter_size) - gaussian filter
    """
    gaus_filter = np.asarray([[1]])
    if filter_size == 1:
        return gaus_filter
    for i in range(filter_size - 1):
        gaus_filter = convolve2d(gaus_filter, GAUSSIAN_KERNEL)
    return np.asarray(gaus_filter)/np.sum(gaus_filter)


def reduce_image(im, filter_vec):
    """
    Blurs given image and samples every second pixel in each row and each column.
    :param im: Original image.
    :param filter_vec: Gaussian vector that is used to blur an image.
    :return: Reduced image.
    """
    image = convolve(convolve(im, filter_vec), filter_vec.T)
    return image[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Gaussian pyramid pyramid of a given image.
    :param im: A grayscale image with double values in [0, 1].
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: pyr - a list of gaussian pyramid
             filter_vec - the gaussian filter that was used.
    """
    pyr = [im]
    filter_vec = get_gaus_filter(filter_size)
    for i in range(1, max_levels):
        pyr.append(np.asarray(reduce_image(pyr[i-1], filter_vec)))
        if (pyr[i].shape[0] < MIN_SHAPE_SIZE*2) or (pyr[i].shape[1] < MIN_SHAPE_SIZE*2):
            break
    return pyr, filter_vec


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
