import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.models import Model
# from keras.preprocessing.image import load_img
# from keras.applications.vgg19 import decode_predictions
#
# from scripts.images import load_and_process_img, imshow
# from model import get_model
from scripts.images import imshow, load_image
from model.index import reshape_and_normalize_image, save_image, generate_noise_image
from model.index import total_cost, compute_content_cost, compute_style_cost_for_layer

# sess = tf.Session()
#
# content_path = './data/sandwhich.jpeg'
# style_path = './data/van-gogh.jpg'
#
# # Initialize Images
# content_image = load_image(content_path)
# content_image = reshape_and_normalize_image(content_image)
#
# style_image = load_image(style_path)
# style_image = reshape_and_normalize_image(style_image)
#
# generated_image = generate_noise_image(content_image)
#
# # Get VGG19 to get initial a_S and a_G
# vgg = VGG19(include_top=True, weights='imagenet')
# model = Model(inputs=vgg.input, outputs=vgg.output)
#
# # sess.run(model['input'].assign(content_image))
# out = model.get_layer('block4_conv2')
#
# a_C = sess.run(out)
#
# a_G = out
#
# J_content = compute_content_cost(a_C, a_G)
