import tensorflow as tf
import numpy as np

from index import *


def test_compute_content_cost():
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
        print("\nJ_content = " + str(J_content.eval()))
        print("Should be = 6.76559\n")


def test_gram_matrix():
    tf.reset_default_graph()

    answer = [[ 6.42230511, -4.42912197, -2.09668207],
              [ -4.42912197, 19.46583748, 19.56387138],
              [ -2.09668207, 19.56387138, 20.6864624 ]]

    with tf.Session() as test:
        tf.set_random_seed(1)
        A = tf.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
        print('Gram Matrix:')
        for r in GA.eval():
            print('\t', r)
        print('Should be:')
        for r in answer:
            print('\t', r)
        print()


def test_compute_style_cost_for_layer():
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = compute_style_cost_for_layer(a_S, a_G)

        print("J_style_layer = " + str(J_style_layer.eval()))
        print("Should be: 9.19028\n")


def test_total_cost():
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(3)
        J_content = np.random.randn()
        J_style = np.random.randn()
        J = total_cost(J_content, J_style)
        print("J =        " + str(J))
        print("Should be: 35.34667875478276\n")


if __name__ == '__main__':
    test_compute_content_cost()
    test_gram_matrix()
    test_compute_style_cost_for_layer()
    test_total_cost()
