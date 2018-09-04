import tensorflow as tf
import keras
from keras.models import Model
from keras.applications.vgg19 import VGG19

DEFAULT_CONTENT_LAYERS = ['block5_conv2']
DEFAULT_STYLE_LAYERS = [('block1_conv1', 0.2),
                        ('block2_conv1', 0.2),
                        ('block3_conv1', 0.2),
                        ('block4_conv1', 0.2),
                        ('block5_conv1', 0.2)]


# def get_model(content_layers=DEFAULT_CONTENT_LAYERS, style_layers=DEFAULT_STYLE_LAYERS):
#     """
#     Take an input image and output content + style layers
#     """
#     # Load VGG19
#     vgg = VGG19(include_top=True, weights='imagenet')
#     vgg.trainable = False
#     # grab output layers
#     content_output = [vgg.get_layer(layer).output for layer in content_layers]
#     style_output = [vgg.get_layer(layer).output for layer,weight in style_layers]
#     model_outputs = content_output + style_output
#     # build model
#     return Model(inputs=vgg.input, outputs=model_outputs)


# Content Loss
def get_content_cost(a_C, a_G):
    """
    Computes the content cost

        1 / (4 * n_H * n_W * n_C) * sum[ (a_C - a_C)^2 ]

    Arguments
        a_C => tensor of dimensions (1, n_H, n_W, n_C) hidden layer activations representing content of the image C
        a_G => tensor of dimensions (1, n_H, n_W, n_C) hidden layer activations representing content of the image G

    Returns
        J_content => scalar
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # reshape 3D => 2D
    a_C_reshaped = tf.reshape(a_C, (n_H * n_W, n_C))
    a_G_reshaped = tf.reshape(a_G, (n_H * n_W, n_C))
    return 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(a_C - a_G))


# Style Loss
def gram_matrix(A):
    """
    Argument:
        A -- matrix of shape (n_C, n_H*n_W)

    Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    return tf.matmul(A, tf.transpose(A))

def get_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H * n_W])

    GS = gram_matrix(a_S)
    GA = gram_matrix(a_G)

    cost_sum = tf.reduce_sum(tf.square(GS - GA))
    return 1 / (4 * (n_C**2) * (n_H * n_W)**2) * cost_sum


# def get_style_loss(base, target):
#     """
#         Style Loss = Sum ()
#     """
#     n_H, n_W, n_C = base.get_shape().as_list()
#     GBase = gram_matrix(base)
#     loss_sum = tf.reduce_mean(tf.square(GBase - target))
#     return 1/(4 * n_C**2 * (n_H * n_W)**2) * loss_sum
#
#
# def get_loss(content_base, content_target, style_base, style_loss):
#     J_content = get_content_loss(content_base, content_target)
#     J_style = get_style_loss(style_base, style_loss)
#     return J_content + J_style
