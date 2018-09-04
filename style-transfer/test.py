import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import decode_predictions
# from keras.applications.vgg19 import preprocess_input

from images import load_and_process_img, imshow
from model import get_model

path_to_img = './data/sandwhich.jpeg'

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
image = load_and_process_img(path_to_img)
model = get_model(content_layers, style_layers)

y_hat = model.predict(image)
print('Predictions:')
for pred in decode_predictions(y_hat)[0]:
    print('\t', pred)

imshow(path_to_img)
plt.show()
