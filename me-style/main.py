import tensorflow as tf
import numpy as np

from helpers.image import get_image, generate_random_image
from helpers.model import get_vgg_model, get_features_from_model

from config import CONTENT_LAYER, STYLE_LAYERS

""" Tensorflow + Keras """
sess = tf.InteractiveSession()



""" Get Images """
content_path = './data/turtle.png'
style_path = './data/van-gogh.jpg'

content_image = get_image(content_path)
style_image = get_image(style_path)
generated_image = generate_random_image()



""" Get VGG19 """
vgg = get_vgg_model()
sess.run(vgg.)
