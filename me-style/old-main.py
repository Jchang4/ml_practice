import tensorflow as tf

from helpers.image import get_image, generate_random_image
from helpers.model import get_vgg_model, get_features_from_model
from helpers.model import compute_content_cost, compute_style_cost
from helpers.model import total_cost

from config import CONTENT_LAYER, STYLE_LAYERS


""" Get Images """
content_path = './data/turtle.png'
style_path = './data/van-gogh.jpg'

content_image = get_image(content_path)
style_image = get_image(style_path)
generated_image = generate_random_image()


""" Get VGG Model """
model = get_vgg_model()
pred = model.predict(content_image)


""" Setup Tensorflow """
sess = tf.Session()


""" Get Activations and initial Cost, J """
a_C = get_features_from_model(model, CONTENT_LAYER)
a_G = get_features_from_model(model, CONTENT_LAYER)
J_content = compute_content_cost(a_C, a_G)
J_style = compute_style_cost(sess, model, STYLE_LAYERS)
J = total_cost(J_content, J_style, alpha=10, beta=40)

""" Start training! """
# Assign the content image to be the input of the VGG model.

optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)

model.compile(loss='mean_squared_error', metrics=['accuracy'])

model.fit(generated_image)

# model_nn(sess, generated_image)
