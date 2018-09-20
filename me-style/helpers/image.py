import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

def show_image(img, title = ''):
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()

def vgg19_preprocess(img):
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def get_image(path_to_img, expected_img_size = (200, 200, 3), show = False, title = ''):
    x = load_img(path_to_img, target_size=expected_img_size)
    if show:
        show_image(x, title)
    x = img_to_array(x).astype('float32')
    x = vgg19_preprocess(x)
    return x

def generate_noise_image(path_to_content_img, expected_img_size, noise_ratio = 0.6, show = True):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    content_image = load_img(path_to_content_img, target_size=expected_img_size)
    content_image = img_to_array(content_image)
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, expected_img_size).astype('float32')
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    if show:
        show_image(input_image, 'Generated Image')
    input_image = vgg19_preprocess(input_image)
    return input_image

def generate_noise_image2(content_image, expected_img_size, noise_ratio = 0.6, show = True):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, expected_img_size).astype('float32')
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    if show:
        show_image(input_image, 'Generated Image')
    input_image = vgg19_preprocess(input_image)
    return input_image

def deprocess_img(img):
    x = img.copy()

    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocess_img()")

    # Inverse of preprocessing
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    # Ensure we're in the range of (0, 255)
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(path_to_save, img):
    img = deprocess_img(img)
    img = Image.fromarray(img)
    img.save(path_to_save, "JPEG")
    