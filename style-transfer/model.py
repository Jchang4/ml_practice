import tensorflow as tf
import keras
from keras.models import Model
from keras.applications.vgg19 import VGG19

DEFAULT_CONTENT_LAYERS = ['block5_conv2']
DEFAULT_STYLE_LAYERS = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']


def get_model(content_layers=DEFAULT_CONTENT_LAYERS, style_layers=DEFAULT_STYLE_LAYERS):
    """
    Take an input image and output content + style layers
    """
    # Load VGG19
    vgg = VGG19(include_top=True, weights='imagenet')
    vgg.trainable = False
    # grab output layers
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_outputs = content_output + style_output
    # build model
    print(model_outputs)
    return Model(inputs=vgg.input, outputs=model_outputs)


# Content Loss
def get_content_loss(base, target):
    """
        Content Loss = Sum (base - target)^2
    """
    return tf.reduce_sum(tf.square(base - target))


# Style Loss
def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))

def get_style_loss(base, target):
    """
        Style Loss = Sum ()
    """
    n_H, n_W, n_C = base.get_shape().as_list()
    GBase = gram_matrix(base)
    loss_sum = tf.reduce_mean(tf.square(GBase - target))
    return 1/(4 * n_C**2 * (n_H * n_W)**2) * loss_sum


def get_loss(content_base, content_target, style_base, style_loss):
    J_content = get_content_loss(content_base, content_target)
    J_style = get_style_loss(style_base, style_loss)
    return J_content + J_style
