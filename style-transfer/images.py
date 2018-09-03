import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image


CONTENT_IMAGE_PATH = './data/turtle.png'
STYLE_IMAGE_PATH = './data/van-gogh.jpg'


def load_img(img_path):
    max_dim = 512

    img = Image.open(img_path)

    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    # Converts PIL to Numpy array
    img = kp_image.img_to_array(img)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    # out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)






if __name__ == '__main__':
    content = load_img(CONTENT_IMAGE_PATH).astype('uint8')
    style = load_img(STYLE_IMAGE_PATH).astype('uint8')

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.show()
