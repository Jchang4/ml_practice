import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import VGG19, decode_predictions
# from keras.applications.vgg19 import preprocess_input

from scripts.images import load_and_process_img, imshow
from model import get_model

get_model()
vgg = VGG19(include_top=True, weights='imagenet')
