import tensorflow as tf
import keras.backend as K

from helpers.image import get_image, save_image, generate_noise_image
from helpers.model import get_model, nst_model

from config import CONFIG

# Reset the graph
tf.reset_default_graph()

with tf.Session() as sess:
    # Get Images
    content_path = './data/oppossum.jpeg'
    style_path = './data/fauvism-1.jpg'

    content_img = get_image(content_path, show=False, title='Content Image')
    style_img = get_image(style_path, show=False, title='Style Image')
    generated_img = generate_noise_image(content_path, show=False)
    
    # Turn images into Tensorflow Constants and Variables
    content_img = K.constant(content_img, name='content_img', dtype=tf.float32)
    style_img = K.constant(style_img, name='style_img', dtype=tf.float32)
    generated_img = K.variable(generated_img, name='generated_img', dtype=tf.float32)

    # Setup Model and Initial Costs
    model = get_model()

    # Train the Model!
    nst_model(sess, model, generated_img, content_img, style_img, save_dir='./images/main')
