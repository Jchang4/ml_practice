import tensorflow as tf

from helpers.image import get_image, generate_random_image
from helpers.model import get_vgg_model, get_features_from_model
from helpers.model import compute_content_cost, compute_style_cost
from helpers.model import total_cost


# Get Images
content_path = './data/turtle.png'
style_path = './data/van-gogh.jpg'

content_image = get_image(content_path)
style_image = get_image(style_path)
generated_image = generate_random_image()


# Get VGG Model
model = get_vgg_model()
pred = model.predict(content_image)

# Setup Tensorflow
sess = tf.Session()

# Get Activations and initial Cost, J
a_C = 0
a_S = 0
a_G = 0
J_content = compute_content_cost(a_C, a_G)
J_style = compute_style_cost()
J = total_cost(J_content, J_style, alpha=10, beta=40)

optimizer = tf.train.AdamOptimizer(2.0)
# train_step = optimizer.minimize(J)
