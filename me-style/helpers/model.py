import tensorflow as tf
from keras.models import Model
from keras.applications.vgg19 import VGG19

from config import TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS


# VGG19
def get_vgg_model():
    """
    Return VGG19 Model
    """
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS))
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.output)

def get_features_from_model(model, layer_name):
    return model.get_layer(layer_name).output


# Cost Functions
def compute_content_cost(a_C, a_G):
    """ Mean Squared Error between
        Style and Generated Image activations

        1 / (4 * height * width * channels) * sum[ (a_C - a_G)^2 ]

        Arguments:
            a_C => matrix of shape (1, height, width, channels)
            a_G => matrix of shape (1, height, width, channels)

        Returns:
            J_content => scalar
    """
    m, height, width, num_channels = a_G.get_shape().as_list()

    constant = 1 / (4 * height * width * num_channels)
    cost_sum = tf.reduce_sum(tf.square(a_C - a_G))

    return constant * cost_sum

def compute_style_cost_one_layer(a_S, a_G):
    pass

def compute_style_cost(layers):
    pass

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha * J_content + beta * J_style
