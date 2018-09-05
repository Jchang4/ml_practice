import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

from config import TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS


# Pre-Process Image
def get_image(path_to_img, show = False):
    """ Load Image as a Numpy Matrix
        with shape (1, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS)

        Returns (1, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS) matrix
    """
    img = image.load_img(path_to_img, target_size=(TARGET_HEIGHT, TARGET_WIDTH))

    if show:
        show_image(img)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

# Display and Save Images
def show_image(image, title = ''):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.show()

def save_image(image, path_to_save):
    pass


# Generate a Random Image!
def generate_random_image(show = False):
    img = np.random.randn(1, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS)
    img *= 255.
    if show:
        show_image(img[0], 'Randomly Generated Image')
    return preprocess_input(img)
