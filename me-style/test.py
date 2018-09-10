import tensorflow as tf

from helpers.image import get_image, save_image, generate_noise_image
from helpers.model import get_model, nst_model

from config import CONFIG

# Reset the graph
tf.reset_default_graph()


""" Get Images """
content_path = './data/coffee.jpeg'
style_path = './data/van-gogh.jpg'

content_img = tf.constant(get_image(content_path, show=False, title='Content Image'), name='content_img', dtype=tf.float32)
style_img = tf.constant(get_image(style_path, show=False, title='Style` Image'), name='style_img', dtype=tf.float32)
generated_img = tf.Variable(generate_noise_image(content_path, show=False), name='generated_img', dtype=tf.float32)

""" Setup Model and Initial Costs """
model = get_model()

sess = tf.Session()

""" Train the Model! """
nst_model(sess, model, generated_img, content_img, style_img)
