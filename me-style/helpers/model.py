import tensorflow as tf
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.applications.vgg19 import VGG19
from helpers.image import save_image

from config import CONFIG


def get_model():
    vgg = VGG19(include_top=False, weights='imagenet',
                input_shape=(CONFIG.HEIGHT, CONFIG.WIDTH, CONFIG.NUM_CHANNELS))
    vgg.trainable = False

    content_layers = [vgg.get_layer(layer_name).output for layer_name in CONFIG.CONTENT_LAYERS]
    style_layers = [vgg.get_layer(layer_name).output for layer_name,coeff in CONFIG.STYLE_LAYERS]
    output_layers = content_layers + style_layers

    return Model(inputs=vgg.input, outputs=output_layers)


""" Cost Function Helpers """
def mean_square_error(x, y):
    return tf.reduce_sum(tf.square(x - y))

def gram_matrix(x):
    return tf.matmul(x, tf.transpose(x))

""" Cost Functions """
def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    constant = 1 / (4. * n_H * n_W * n_C)
    return constant * mean_square_error(a_C, a_G)

def compute_style_cost_layer(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H * n_W])

    gram_S = gram_matrix(a_S)
    gram_G = gram_matrix(a_G)

    constant = 1 / (4. * n_C**2 * (n_H * n_W)**2)
    return constant * mean_square_error(gram_S, gram_G)

def compute_style_cost(style_activations, generated_activations):
    J_style = 0
    for i,(layer_name,coeff) in enumerate(CONFIG.STYLE_LAYERS):
        a_S = style_activations[i]
        a_G = generated_activations[i]
        J_style += coeff * compute_style_cost_layer(a_S, a_G)
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha * J_content + beta * J_style

""" Compute Gradient """
def compute_gradient(loss, img):
    return K.gradients(loss, img)


""" NST Model """
def nst_model(sess, model, input_img, content_img, style_img,
                num_iterations = 200, alpha = 0.01):
    a_C = model(content_img)[0]
    a_S = model(style_img)[1:]
    a_G = model(input_img)

    # Cost
    J_content = compute_content_cost(a_C, a_G[0])
    J_style = compute_style_cost(a_S, a_G[1:])
    J = total_cost(J_content, J_style)

    optimizer = tf.train.AdamOptimizer(0.01)
    grads = compute_gradient(J, input_img)[0]
    train_step = optimizer.apply_gradients([(grads, input_img)])
    # train_step = optimizer.minimize(J)

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    sess.run(tf.global_variables_initializer())

    for i in range(num_iterations):
        sess.run(train_step)

        clipped = tf.clip_by_value(input_img, min_vals, max_vals)
        input_img.assign(clipped)

        # Print and Save Image every 20 interations
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print('Total Cost:  ', Jt)
            print('Content Cost:', Jc)
            print('Style Cost:  ', Js, '\n')
            save_image('./images/cli/nst-{}.jpg'.format(str(i)), sess.run(input_img)[0])

    save_image('./images/cli/nst-final.jpg', sess.run(input_img)[0])
