import tensorflow as tf
import keras
from keras.models import Model
from keras.applications.vgg19 import VGG19

def get_model(content_layers, style_layers):
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
    # return Model(inputs=vgg.input, outputs=model_outputs)
    return Model(inputs=vgg.input, outputs=vgg.output)

def get_content_loss(base, target):
    """
        Content Loss = Sum (base - target)^2
    """
    return tf.reduce_sum(tf.square(base - target))

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



def get_loss():
    pass
