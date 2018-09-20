import tensorflow as tf
import keras.backend as K

from helpers.image import get_image, save_image, generate_noise_image
from helpers.model import get_model, nst_model

tfe = tf.contrib.eager

tf.enable_eager_execution()

class CONFIG:
    # Image
    HEIGHT = 300
    WIDTH = 400
    NUM_CHANNELS = 3

    # Layers for Cost Function
    CONTENT_LAYERS = ['block5_conv2']
    STYLE_LAYERS = [
                    # ('block1_conv1', 0.2),
                    ('block2_conv1', 0.2),
                    ('block3_conv1', 0.2),
                    ('block4_conv1', 0.2),
                    # ('block5_conv1', 0.2)
    ]

IMAGE_SIZE = (CONFIG.HEIGHT, CONFIG.WIDTH, CONFIG.NUM_CHANNELS)

# Get Images
content_path = './data/content/crazy-giuliani.jpeg'
style_path = './data/style/psychadelic-1.jpg'

content_img = get_image(content_path, IMAGE_SIZE, show=False, title='Content Image')
style_img = get_image(style_path, IMAGE_SIZE, show=False, title='Style Image')
generated_img = generate_noise_image(content_path, IMAGE_SIZE, show=False)

# Turn images into Tensorflow Constants and Variables
content_img = tf.constant(content_img, name='content_img', dtype=tf.float32)
style_img = tf.constant(style_img, name='style_img', dtype=tf.float32)
generated_img = tfe.Variable(generated_img, name='generated_img', dtype=tf.float32)

# Setup Model and Initial Costs
model = get_model(IMAGE_SIZE, CONFIG.CONTENT_LAYERS, CONFIG.STYLE_LAYERS)

# Train the Model!
nst_model(model, generated_img, content_img, style_img, CONFIG.STYLE_LAYERS,
            save_dir='./images/main',
            alpha = 10, beta = 40)
