import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.applications.vgg19 import VGG19
from tensorflow.python.keras.preprocessing import image as kp_image
from keras.preprocessing.image import load_img

DEFAULT_CONTENT_LAYERS = ['block5_conv2']
DEFAULT_STYLE_LAYERS = [('block1_conv1', 0.2),
                        ('block2_conv1', 0.2),
                        ('block3_conv1', 0.2),
                        ('block4_conv1', 0.2),
                        ('block5_conv1', 0.2)]
class CONFIG:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


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
def compute_content_cost(a_C, a_G):
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

def compute_style_cost_for_layer(a_S, a_G):
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
    GG = gram_matrix(a_G)

    cost_sum = tf.reduce_sum(tf.square(GS - GG))
    return 1 / (4 * (n_C**2) * (n_H * n_W)**2) * cost_sum

def compute_style_cost(model, STYLE_LAYERS = DEFAULT_STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]
        # Set a_S to be the hidden layer activation from the layer
        # we have selected, by running the session on out
        a_S = sess.run(out)
        # Set a_G to be the hidden layer activation from same layer.
        # Here, a_G references model[layer_name] and isn't evaluated yet.
        # Later in the code, we'll assign the image G as the model input,
        # so that when we run the session, this will be the activations
        # drawn from the appropriate layer, with G as input.
        a_G = out
        # Compute style_cost for the current layer
        J_style_layer = compute_style_cost_for_layer(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

# Combination of Content and Style Cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))

    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS

    return image

def save_image(path, image):

    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image

def load_image(path_to_img):
    return load_img(path_to_img, target_size=(224, 224))

def load_and_process_img(path_to_img):
  img = load_image(path_to_img)
  img = kp_image.img_to_array(img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  img = np.expand_dims(img, axis=0)
  return img
