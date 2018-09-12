import tensorflow as tf

from helpers.image import get_image, save_image, generate_noise_image
from helpers.model import get_model, nst_model

from config import CONFIG

# Reset the graph
tf.reset_default_graph()

with tf.Session() as sess:
    # Get Images
    content_path = './data/seal.jpeg'
    style_path = './data/cubism-1.jpg'

    content_img = tf.constant(get_image(content_path, show=False, title='Content Image'), name='content_img', dtype=tf.float32)
    style_img = tf.constant(get_image(style_path, show=False, title='Style` Image'), name='style_img', dtype=tf.float32)
    generated_img = tf.Variable(generate_noise_image(content_path, show=False), name='generated_img', dtype=tf.float32)

    # Setup Model and Initial Costs
    model = get_model()

    # Train the Model!
    nst_model(sess, model, generated_img, content_img, style_img)
