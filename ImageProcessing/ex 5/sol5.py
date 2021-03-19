import skimage.color as sk
import imageio
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
from . import sol5_utils

BITS = 256


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation: RGB or gray-scale.
    :param filename: A path to an image that we want to work on, string
    :param representation: Tells in which representation the output should be: 1 for gray-scale, 2 for RGB.
    :return: Image in given representation
    """
    image = imageio.imread(filename)
    is_gray = True  # indicates if the image is grays-cale or not
    image_float = (image / (BITS - 1)).astype(np.float64)
    if len(image.shape) > 2 and image.shape[-1] == 3:  # image is RGB
        is_gray = False
    if (not is_gray) and (representation == 1):
        return sk.rgb2gray(image_float)
    elif (is_gray and (representation == 1)) or representation == 2:
        return image_float


def crop_im(im_1, crop_size,  im_2=None, single=True):
    """
    This function randomly crops given image with given size. If there are a couple of images to crop, it crops them
    both in the same way
    :param im_1: Image to crop.
    :param im_2: Second image to crop.
    :param crop_size: A tuple (height, width) specifying the crop size of the image
    :param single: Indicator that tells if there is a couple of images to crop or a single image.
    :return: Cropped image if there is a single image to crop and touple of 2 images cropped if there are 2 images to
             crop.
    """
    im_h, im_w = im_1.shape
    top = np.random.choice(im_h - crop_size[0])
    left = np.random.choice(im_w - crop_size[1])
    if single:
        return im_1[top:top+crop_size[0], left:left+crop_size[1]]
    return im_1[top:top+crop_size[0], left:left+crop_size[1]], im_2[top:top+crop_size[0], left:left+crop_size[1]]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Generator that generates pairs of patches comprising (i) an original, clean and sharp,
    image patch with (ii) a corrupted version of same patch.
    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
                            and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: data_generator, a Python’s generator object which outputs random tuples of the form
             (source_batch, target_batch), where each output variable is an array of shape
             (batch_size, height, width, 1), target_batch is made of clean images, and source_batch is their respective
             randomly corrupted version according to corruption_func(im)
    """
    images_cache = {}

    def generator():
        """
        Generator that generates pairs of patches comprising (i) an original, clean and sharp,
        image patch with (ii) a corrupted version of same patch.
        :return: tuples of the form (source_batch, target_batch).
        """
        height, width = crop_size
        source_batch = np.zeros((batch_size, height, width, 1), dtype=np.float64)
        target_batch = np.zeros((batch_size, height, width, 1), dtype=np.float64)
        while True:

            for i in range(batch_size):
                filename = np.random.choice(filenames)

                if filename not in images_cache:
                    images_cache[filename] = read_image(filename, 1)

                im = images_cache[filename]
                clean_im = crop_im(im, (3*crop_size[0], 3*crop_size[1]))
                corrupted_im = corruption_func(clean_im)
                target, source = crop_im(clean_im, crop_size, corrupted_im, False)
                target_batch[i, :, :, 0] = target - 0.5
                source_batch[i, :, :, 0] = source - 0.5
            yield (source_batch, target_batch)

    return generator()


def resblock(input_tensor, num_channels):
    """
    The function takes as input a symbolic input tensor and the number of channels for each of its
    convolution layers, and returns the symbolic output tensor of the layer configuration described above.
    :param input_tensor: Input tensor.
    :param num_channels: Number of channels for each of its convolutional layers.
    :return: Residual output tensor.
    """
    output_tensor = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Conv2D(num_channels, (3, 3), padding='same')(output_tensor)
    output_tensor = Add()([input_tensor, output_tensor])
    output_tensor = Activation('relu')(output_tensor)
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Builds untrained Keras model.
    :param height: Height of the input tensor.
    :param width: Width of the input tensor.
    :param num_channels: Number of channels for each of its convolutional layers.
    :param num_res_blocks: Number of residual blocks
    :return: Untrained Keras model.
    """
    input_tensor = Input((height, width, 1))
    output_tensor = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    output_tensor = Activation('relu')(output_tensor)
    for i in range(num_res_blocks):
        output_tensor = resblock(output_tensor, num_channels)
    output_tensor = Conv2D(1, (3, 3), padding='same')(output_tensor)
    output_tensor = Add()([input_tensor, output_tensor])
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Trains network model.
    :param model: A general neural network model for image restoration
    :param images: A list of file paths pointing to image files. You should assume these paths are complete, and
                   should append anything to them.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
                            and returns a randomly corrupted version of the input image.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    :return: Trained model
    """
    shuffled = images
    np.random.shuffle(shuffled)
    split_ind = int(len(shuffled)*0.8)
    input_shape = model.input_shape[1:3]
    train_set_generator = load_dataset(images[:split_ind], batch_size, corruption_func, input_shape)
    valid_set_generator = load_dataset(images[split_ind:], batch_size, corruption_func, input_shape)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_set_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=valid_set_generator, validation_steps=num_valid_samples)
    return model


def restore_image(corrupted_image, base_model):
    """
    Creates a new model that fits the size of the input image and has the same weights as the given base model.
    :param corrupted_image: A grayscale image of shape (height, width) and with values in the [0, 1] range of
                            type float64, that is affected by a corruption generated from corruption function.
    :param base_model: Neural network trained to restore small patches
    :return:Restored image
    """
    h, w = corrupted_image.shape
    input_tensor = Input((h, w, 1))
    output_tensor = base_model(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.set_weights(base_model.get_weights())
    predicted = model.predict(corrupted_image[np.newaxis, ..., np.newaxis] - 0.5)
    restored_image = (np.clip(predicted + 0.5, 0, 1).astype(np.float64))[0, :, :, 0]
    return restored_image


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Adds random zero mean gaussian noise to given image.
    :param image: A grayscale image with values in the [0, 1] range of type float64
    :param min_sigma: A non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: A non-negative scalar value larger than or equal to min_sigma, representing the maximal
                      variance of the gaussian distribution.
    :return: Corrupted with gaussian noise image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    gaussian_moise = np.random.normal(0, sigma, image.shape)
    corrupted_im = np.float64(np.clip(np.rint(255 * (image + gaussian_moise))/255, 0, 1))
    return corrupted_im


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Trains denoising model.
    :param num_res_blocks: Number of residual blocks.
    :param quick_mode: Quick training for model.
    :return: Trained model for denoising.
    """
    def corruption_func(image):
        return add_gaussian_noise(image, 0, 0.2)
    images = sol5_utils.images_for_denoising()
    size = 24
    channels = 48
    model = build_nn_model(size, size, channels, num_res_blocks)
    if quick_mode:
        train_model(model, images, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images, corruption_func, 100, 100, 5, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Adds motion blur to given image.
    :param image: A grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: An odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: An angle in radians in the range [0, π).
    :return: Corrupted image with motion blur.
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Adds random motion blur to given image.
    :param image: A grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: A list of odd integers.
    :return: Randomly corrupted image with motion blur.
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, kernel_size, angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    Trains deblurring model.
    :param num_res_blocks: Number of residual blocks.
    :param quick_mode: Quick training for model.
    :return: Trained model for debluring.
    """
    def corruption_func(image):
        return random_motion_blur(image, [7])
    images = sol5_utils.images_for_deblurring()
    size = 16
    channels = 32
    model = build_nn_model(size, size, channels, num_res_blocks)
    if quick_mode:
        train_model(model, images, corruption_func, 10, 3, 2, 30)
    else:
        train_model(model, images, corruption_func, 100, 100, 10, 1000)
    return model
