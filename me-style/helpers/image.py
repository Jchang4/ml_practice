import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

# from PIL import Image


# Configurable Constants
TARGET_HEIGHT = 512
TARGET_WIDTH = 512
TARGET_CHANNELS = 3


# Pre-Process Image
def get_image(path_to_img, show = False):
    """ Load Image as a Numpy Matrix
        with shape (1, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS)
    """
    img = image.load_img(path_to_img, target_size=(TARGET_HEIGHT, TARGET_WIDTH))

    if show:
        show_image(img)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

def vgg_preprocess(img_matrix):
    return preprocess_input(img_matrix)


# Display and Save Images
def show_image(image, title = ''):
    plt.figure(figsize=(TARGET_HEIGHT,TARGET_WIDTH))
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.show()

def save_image(image, path_to_save):
    pass
