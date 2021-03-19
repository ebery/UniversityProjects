import numpy as np
import imageio
import skimage.color as sk
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os


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
        pyr.append(reduce_image(pyr[i-1], filter_vec))
        if (pyr[i].shape[0] < MIN_SHAPE_SIZE*2) or (pyr[i].shape[1] < MIN_SHAPE_SIZE*2):
            break
    return pyr, filter_vec


def expand_image(im, filter_vec):
    """
    Up samples an image by adding 0 in the odd places.
    :param im: image to expand.
    :param filter_vec: Gaussian vector that is used to blur an image.
    :return: Expanded image.
    """
    m, n = im.shape
    exp_filter = 2*filter_vec
    expended_im = np.zeros((2*m, 2*n), dtype=im.dtype)
    expended_im[::2, ::2] = im
    return convolve(convolve(expended_im, exp_filter), exp_filter.T)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Laplacian pyramid pyramid of a given image.
    :param im: A grayscale image with double values in [0, 1].
    :param max_levels: The maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: pyr - a list of laplacian pyramid
             filter_vec - the gaussian filter that was used.
    """
    pyr = []
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(gaussian_pyramid) - 1):
        m, n = gaussian_pyramid[i].shape
        pyr.append(gaussian_pyramid[i]-expand_image(gaussian_pyramid[i+1], filter_vec)[:m, :n])
    pyr.append(gaussian_pyramid[-1])  # add the smallest image
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Constructs the original image from laplacian pyramid.
    :param lpyr: Laplacian pyramid.
    :param filter_vec: Gaussian filter.
    :param coeff: A python list. The list length is the same as the number of levels in the pyramid lpyr.
    :return: Constructed image.
    """
    reversed_lpyr = lpyr[::-1]  # to start from the smallest one
    reversed_coeff = coeff[::-1]
    im = expand_image(np.multiply(reversed_lpyr[0], reversed_coeff[0]), filter_vec) + \
                      np.multiply(reversed_lpyr[1], reversed_coeff[1])
    for i in range(2, len(lpyr)):
        im = expand_image(im, filter_vec) + np.multiply(reversed_lpyr[i], reversed_coeff[i])
    return im


def render_pyramid(pyr, levels):
    """
    Stacks the wanted number of pyramid levels in one image with black background.
    :param pyr: Either a Gaussian or Laplacian pyramid.
    :param levels: Number of levels to present in the result.
    :return: A single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally.
    """
    m, n = pyr[0].shape
    for i in range(levels - 1):
        n += pyr[i+1].shape[1]
    # we want to represent black as the minimal value of all the pyramid images.
    # I did it because I noticed that stretching changes the gray level of smaller images in the
    # pyramids
    res = np.full((m, n), np.min(np.concatenate([[np.min(pyr[i]) for i in range(levels)], [0]])))
    col_start = 0
    for i in range(levels):
        res[:pyr[i].shape[0], col_start:col_start+pyr[i].shape[1]] = pyr[i]
        col_start += pyr[i].shape[1]
    return res


def display_pyramid(pyr, levels):
    """
    Displays the pyramid images stacked horizontally on black background
    :param pyr: Either a Gaussian or Laplacian pyramid.
    :param levels: Number of levels to present in the result.
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.axis('off')
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    The function is performing a blending between 2 input images using input mask.
    :param im1: First gayscale image to be blended.
    :param im2: Second gayscale image to be blended.
    :param mask: A boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                 of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: The max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: The size of the Gaussian filter (an odd scalar that represents a squared filter) which
                           defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: The size of the Gaussian filter(an odd scalar that represents a squared filter) which
                             defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: Blended image.
    """
    L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    Gm = build_gaussian_pyramid(np.asarray(mask, dtype=np.float64), max_levels, filter_size_mask)[0]
    Lout = [(np.multiply((1-Gm[i]), L1[i]) + np.multiply(Gm[i], L2[i])) for i in range(max_levels)]
    coeff = [1]*max_levels
    im_blend = laplacian_to_image(Lout, filter_vec, coeff)
    return np.clip(im_blend, 0, 1)


def blending_example1():
    """
    The first example of blending
    :return: im1 - First image for blending.
             im2 - Second image for blending.
             mask - The mask that was used for blending.
             im_blend - blended image.
    """
    im1 = read_image(relpath('externals/eagle.jpg'), 2)
    im2 = read_image(relpath('externals/owl.jpg'), 2)
    mask = read_image(relpath('externals/eagle_mask.jpg'), 1)
    m, n = mask.shape
    mask = mask < 1
    im_blend = np.zeros((m, n, 3))
    for i in range(3):
        im_blend[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 4, 3, 3)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(im1)
    ax[0, 0].title.set_text('im1')
    ax[0, 1].imshow(im2)
    ax[0, 1].title.set_text('im2')
    ax[1, 0].imshow(mask, cmap='gray')
    ax[1, 0].title.set_text('mask')
    ax[1, 1].imshow(im_blend)
    ax[1, 1].title.set_text('blend')
    # plt.show()
    return im1, im2, mask, im_blend


def blending_example2():
    """
    The second example of blending
    :return: im1 - First image for blending.
             im2 - Second image for blending.
             mask - The mask that was used for blending.
             im_blend - blended image.
    """
    im1 = read_image(relpath('externals/galaxy.jpg'), 2)
    im2 = read_image(relpath('externals/dog.jpg'), 2)
    mask = read_image(relpath('externals/dog_mask.jpg'), 1)
    m, n = mask.shape
    mask = mask < 1
    im_blend = np.zeros((m, n, 3))
    for i in range(3):
        im_blend[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 4, 3, 3)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(im1)
    ax[0, 0].title.set_text('im1')
    ax[0, 1].imshow(im2)
    ax[0, 1].title.set_text('im2')
    ax[1, 0].imshow(mask, cmap='gray')
    ax[1, 0].title.set_text('mask')
    ax[1, 1].imshow(im_blend)
    ax[1, 1].title.set_text('blend')
    # plt.show()
    return im1, im2, mask, im_blend


def relpath(filename):
    """
    The function returns relative path of the file.
    :param filename: Name of the file.
    :return: Reltive path.
    """
    return os.path.join(os.path.dirname(__file__), filename)


blending_example1()
blending_example2()
plt.show()
