import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image as kp_image
from keras.preprocessing.image import load_img



CONTENT_IMAGE_PATH = './data/turtle.png'
STYLE_IMAGE_PATH = './data/van-gogh.jpg'


def load_image(path_to_img):
    img = load_img(path_to_img, target_size=(224, 224))
    return kp_image.img_to_array(img)

# def load_img(path_to_img):
#     max_dim = 512
#
#     img = Image.open(path_to_img)
#
#     long = max(img.size)
#     scale = max_dim/long
#     img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
#     # Converts PIL to Numpy array
#     img = kp_image.img_to_array(img)
#     # We need to broadcast the image array such that it has a batch dimension
#     img = np.expand_dims(img, axis=0)
    # return img

def imshow(path_to_img, title=None):
    # Remove the batch dimension
    img = load_image(path_to_img)
    # Normalize for display
    if title is not None:
        plt.title(title)
    plt.imshow(img)

def load_and_process_img(path_to_img):
  img = load_image(path_to_img)
  img = kp_image.img_to_array(img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  img = np.expand_dims(img, axis=0)
  return img

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")

  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x





if __name__ == '__main__':
    content = load_img(CONTENT_IMAGE_PATH).astype('uint8')
    style = load_img(STYLE_IMAGE_PATH).astype('uint8')

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.show()
