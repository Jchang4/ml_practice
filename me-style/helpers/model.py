import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from helpers.image import save_image

def get_model(expected_img_size, content_layers, style_layers):
    if len(expected_img_size) != 3:
        raise Exception('Image size must be 3: height, width, and num. of channels')
    
    vgg = VGG19(include_top=False, 
                weights='imagenet',
                input_shape=expected_img_size)
    vgg.trainable = False

    content_layers = [vgg.get_layer(layer_name).output for layer_name in content_layers]
    style_layers = [vgg.get_layer(layer_name).output for layer_name,coeff in style_layers]
    output_layers = content_layers + style_layers

    return Model(vgg.input, output_layers)

""" Cost Function Helpers """
def mean_square_error(x, y):
    return tf.reduce_sum(tf.square(tf.subtract(x, y)))

def gram_matrix(x):
    return tf.matmul(x, tf.transpose(x))

""" Cost Functions """
def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    constant = 1. / (4. * n_H * n_W * n_C)
    return constant * mean_square_error(a_C, a_G)

def compute_style_cost_layer(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H * n_W])

    gram_S = gram_matrix(a_S)
    gram_G = gram_matrix(a_G)

    constant = 1. / (4. * n_C**2 * (n_H * n_W)**2)
    return constant * mean_square_error(gram_S, gram_G)

def compute_style_cost(style_activations, generated_activations, style_layers):
    J_style = 0
    for i,(layer_name,coeff) in enumerate(style_layers):
        a_S = style_activations[i]
        a_G = generated_activations[i]
        J_style += coeff * compute_style_cost_layer(a_S, a_G)
    return J_style

def total_cost(J_content, J_style, alpha, beta):
    return alpha * J_content + beta * J_style

""" Compute Gradient """
def compute_gradient(model, input_img, content_img, style_img,
                    style_layers, alpha, beta):
    with tf.GradientTape(persistent=True) as tape:
        a_C = model(content_img)[0]
        a_S = model(style_img)[1:]
        a_G = model(input_img)
        J_content = compute_content_cost(a_C, a_G[0])
        J_style = compute_style_cost(a_S, a_G[1:], style_layers)
        J_total = total_cost(J_content, J_style, alpha, beta)
    return tape.gradient(J_total, [input_img])[0], J_total, J_content, J_style


""" NST Model """
def nst_model(model, input_img, content_img, style_img, style_layers, 
            num_iterations = 200, alpha = 10, beta = 40, save_dir = './images'):
    # norm_means = np.array([103.939, 116.779, 123.68])
    # min_vals = -norm_means
    # max_vals = 255 - norm_means        
    optimizer = tf.train.AdamOptimizer(2.0)

    abs_start_time = time.time()
    start_time = abs_start_time

    for i in range(num_iterations):
        grads, J_total, J_content, J_style = compute_gradient(model, input_img, content_img, style_img,
                                                                style_layers, alpha, beta)
        optimizer.apply_gradients([(grads, input_img)])
        # input_img = tf.clip_by_value(input_img, min_vals, max_vals)

        # Print and Save Image every 10 interations
        if i % 10 == 0:
            end_time = time.time()
            print('Adding image:', 'nst-{}.jpg'.format(str(i)))
            print('Total Cost:  ', J_total.numpy())
            print('Content Cost:', J_content.numpy())
            print('Style Cost:  ', J_style.numpy())
            save_img_path = '{}/{}.jpeg'.format(save_dir, str(i))
            save_image(save_img_path, input_img.numpy()[0])
            print('Total time:  ', end_time - start_time, '\n')
            start_fn = time.time()

    print('Total time to train: {}'.format(time.time() - start_time))
    save_image('{}/final.jpeg'.format(save_dir), input_img.numpy()[0])
