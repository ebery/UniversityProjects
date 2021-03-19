import numpy as np
import skimage.color as sk
import matplotlib.pyplot as plt
import imageio

BITS = 256
TRANSFORM = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


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


def imdisplay(filename, representation):
    """
    This function displays an image file in given representation: RGB or grayscale.
    :param filename: A path to an image that we want to display, string
    :param representation: Tells in which representation the output should be: 1 for gray-scale, 2 for RGB.
    """
    im_to_display = read_image(filename, representation)
    plt.figure()
    plt.axis('off')
    plt.imshow(im_to_display, cmap="gray")
    plt.show()


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into the YIQ color space.
    :param imRGB: A matrix of RGB image in format float64 normalized to range [0, 1]
    :return: A matrix of YIQ image in format float64 normalized to range [0, 1]
    """
    return np.dot(imRGB, TRANSFORM.T.copy())


def yiq2rgb(imYIQ):
    """
    Transforms an YIQ image into the RGB color space.
    :param imYIQ: A matrix of YIQ image in format float64 normalized to range [0, 1]
    :return: A matrix of RGB image in format float64 normalized to range [0, 1]
    """
    return np.dot(imYIQ, np.linalg.inv(TRANSFORM).T.copy())


def check_rgb(image):
    """
    The function checks if the image is RGB or Gray-scale. If it is Gray-scale then converts to YIQ
    and returns only the Y part of the image. If it is Gray-scale then returns the image as is.
    :param image: The image to check. Range of pixels - [0, 1]
    :return: The Y part of the image if the image is RGB or the image itself if it is Grayscale.
    """
    im_yiq = []
    rgb = False
    y = image
    if len(image.shape) > 2 and image.shape[-1] == 3:  # The image is RGB
        rgb = True
        im_yiq = rgb2yiq(image)  # convert to YIQ format
        y = im_yiq[:, :, 0]
    return rgb, y, im_yiq


def gray2rgb(rgb, y, im_yiq):
    """
    The function checks if the original image was RGB or gray-scale. If it was RGB than it converts it back to RGB.
    Otherwise, returns the image in Gray-scale.
    :param rgb: Indicator that tells if the original image was RGB or ray-scale.
    :param y: Current image.
    :param im_yiq: Original image in YIQ format.
    :return: If the original image was RGB, then returns modified image in YIQ format. Otherwise, it returns current
             image as is in Gray-scale
    """
    if rgb:  # if the original image was RGB, then convert back to RGB format
        im_yiq[:, :, 0] = y/(BITS - 1)
        y = np.clip(yiq2rgb(im_yiq), 0, 1)
    else:
        y = (y/(BITS - 1)).astype(np.float64)
    return y


def histogram_equalize(im_orig):
    """
    The function makes n equalization of the given image and it's histogram.
    :param im_orig: The original normalized matrix of image, RGB or grayscale.
    :return: [im_eq, hist_orig, hist_eq] where:
             im_eq = equalized image
             hist_orig = original histogram
             hist_eq = equalized histogram
    """
    rgb, y, im_yiq = check_rgb(im_orig)
    # The algorithm of histogram equalization starts from here
    hist_orig, bin_edges = np.histogram(y*(BITS - 1), BITS, (0, BITS - 1))  # calculate original histogram
    cum_hist = np.cumsum(hist_orig)
    cum_hist_eq = np.rint(((cum_hist - cum_hist[np.nonzero(cum_hist)[0][0]]) /  # formula from the class
                           (cum_hist[BITS-1] - cum_hist[np.nonzero(cum_hist)[0][0]]))*(BITS-1))
    im_eq = cum_hist_eq[(y*(BITS - 1)).astype(int)]  # get the equalized image
    hist_eq, bin_edges_eq = np.histogram(im_eq, BITS, (0, BITS - 1))  # calculate equalized histogram
    im_eq = gray2rgb(rgb, im_eq, im_yiq)
    if rgb:
        im_eq = np.clip(im_eq, 0, 1)  # to avoid cases when the pixel value is less then 0 or more then 1
    return [im_eq, hist_orig, hist_eq]


def initial_z(hist_orig, n_quant):
    """
    Calculates the first time values of z vector.
    :param hist_orig: histogram of original picture.
    :param n_quant: The number of intensities that the image should have.
    :return: Initial z vector.
    """
    cum_hist = np.cumsum(hist_orig)  # cumulative histogram of original image.
    z = []
    div_pixels = cum_hist[-1] // n_quant  # approximate number of pixels that should be in each division
    for i in range(1, n_quant):
        z_i = np.where(cum_hist > div_pixels*i)[0][0]
        if z_i in z:  # if there is a specific bin with more pixels then before there will be z_i that repeats
            hist_orig[z_i] = 0
            z = initial_z(hist_orig, n_quant - 1).append(z_i)
        else:
            z.append(z_i)
    return z


def find_q(z, hist_orig):
    """
    Find q vector from z vector.
    :param z: z vector.
    :param hist_orig: Histogram of original picture
    :return: q vector calculated from v vector
    """
    gray_levels = np.arange(BITS)  # vector of all the gray levels in range [0, 255]
    q = np.zeros(len(z)-1)
    for i in range(len(z)-1):
        # formula from the class
        q[i] = np.sum(gray_levels[z[i]:z[i+1]+1]*hist_orig[z[i]:z[i+1]+1]) / np.sum(hist_orig[z[i]:z[i+1]+1])
    return q.round().astype(np.uint8)


def find_z(q):
    """
    Find q vector from z vector.
    :param q: q vector.
    :return: z vector calculated from q vector.
    """
    z = np.zeros(len(q) - 1, int)
    for i in range(len(q) - 1):
        z[i] = np.ceil(q[i]/2 + q[i+1]/2)  # formula from class
    return np.concatenate([[0], z, [BITS-1]])


def im_lut(q, z):
    """
    Calculates look-up table of the image for quantization.
    :param q: q vector of the image.
    :param z: z vector of the image.
    :return: Look-up table.
    """
    lut = np.zeros(BITS)
    for i in range(len(q)):
        lut[z[i]:z[i+1]] = q[i]
    lut[-1] = q[-1]  # Handle with the edge
    return lut


def calc_err(q, z, hist_orig):
    """
    Calculates the error of one iteration
    :param q: q vector of the image.
    :param z: z vector of the image.
    :param hist_orig: Histogram of the original image.
    :return: Error for one iteration.
    """
    lut = im_lut(q, z)
    err = np.sum(hist_orig*np.square(lut-np.arange(BITS)))
    return err


def quantize(im_orig, n_quant, n_iter):
    """
    The function calculates quantized image and calculates the error
    :param im_orig: Gray-scale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: Number of intensities that output image should have.
    :param n_iter: Maximum number of iterations of the optimization procedure.
    :return: Quantized image and an array with shape (n_iter,) (or less) of the total intensities error for each
             iteration of the quantization procedure.
    """
    rgb, y, im_yiq = check_rgb(im_orig)
    hist_orig, bin_edges = np.histogram(y*(BITS - 1), BITS, (0, BITS - 1))
    z = np.concatenate([[0], initial_z(hist_orig, n_quant), [(BITS-1)]])
    q = find_q(z, hist_orig)
    error = [calc_err(q, z, hist_orig)]
    for i in range(n_iter - 1):  # first iteration was already done
        z_new = find_z(q)
        if np.array_equal(z_new, z):  # stop if the z and q vectors are already optimal
            break
        q = find_q(z_new, hist_orig)
        z = z_new
        error.append(calc_err(q, z, hist_orig))
    lut = im_lut(q, z)
    im_quant = lut[(y*(BITS - 1)).astype(np.uint8)].astype(np.uint8)  # calculate quantized image.
    im_quant = gray2rgb(rgb, im_quant, im_yiq)  # if the original image was RGB then convert back to RGB
    return im_quant, error


[im_eq, hist_orig] = quantize(read_image('monkey.jpg', 1), 3, 6)

plt.figure()
plt.imshow(read_image('monkey.jpg', 1), cmap='gray')
plt.figure()
plt.imshow(im_eq, cmap='gray')
plt.show()