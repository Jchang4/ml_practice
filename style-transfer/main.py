import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import decode_predictions
# from keras.applications.vgg19 import preprocess_input

from scripts.images import load_and_process_img, imshow
from model import get_model


path_to_img = './data/sandwhich.jpeg'

image = load_and_process_img(path_to_img)
model = get_model()
