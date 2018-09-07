import tensorflow as tf
from keras.models import Model
from keras.applications.vgg19 import VGG19

from config import TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS


# VGG19
def get_vgg_model(output_layer='block5_pool'):
    """
    Return VGG19 Model
    """
    vgg = VGG19(include_top=False,
                weights='imagenet',
                input_shape=(TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS))
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
            a_C => matrix of shape (1, height, width, num_channels)
            a_G => matrix of shape (1, height, width, num_channels)

        Returns:
            J_content => scalar
    """
    m, height, width, num_channels = a_G.get_shape().as_list()

    constant = 1 / (4 * height * width * num_channels)
    cost_sum = tf.reduce_sum(tf.square(a_C - a_G))

    return constant * cost_sum

def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))

def compute_style_cost_one_layer(a_S, a_G):
    """ Style cost for a single layer

        Arguments:
            a_S => matrix of shape (1, height, width, num_channels)
            a_G => matrix of shape (1, height, width, num_channels)
    """
    m, height, width, num_channels = a_G.get_shape().as_list()

    # Change shape to (num_channels, height * width)
    a_S = tf.reshape(tf.transpose(a_S), [num_channels, height * width])
    a_G = tf.reshape(tf.transpose(a_G), [num_channels, height * width])

    S_gram = gram_matrix(a_S)
    G_gram = gram_matrix(a_G)

    constant = 1 / (4 * num_channels**2 * (height * width)**2)
    return constant * tf.reduce_sum(tf.square(S_gram - G_gram))


def compute_style_cost(sess, model, layers):
    J_style = 0
    for layer_name,co_eff in layers:
        out = get_features_from_model(model, layer_name)
        a_S = sess.run(out)
        a_G = out
        J_current_style = compute_style_cost_one_layer(a_S, a_G)
        J_style += co_eff * J_current_style
    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha * J_content + beta * J_style


# Model
def neural_style_transfer_model(sess, input_image, num_iterations=300):
    sess.run(tf.global_variables_initializer())


    print('Transferring Styles... get excited!')
    # for i in num_iterations:
        # Get content cost using a_C, a_G

        # Get style cost using all a_S and sum layers

        # Add content and style costs, taking alpha and beta into account


        # if i % 20 == 0:




    print('You did it! You can find your image here: ./images/' + output_file)
