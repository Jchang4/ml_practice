import tensorflow as tf

from index import *


def test_get_content_cost():
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = get_content_cost(a_C, a_G)
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


def test_get_style_cost():
    tf.reset_default_graph()

    with tf.Session() as test:
        tf.set_random_seed(1)
        a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = get_style_cost(a_S, a_G)

        print("J_style_layer = " + str(J_style_layer.eval()))
        print("Should be: 9.19028\n")


if __name__ == '__main__':
    test_get_content_cost()
    test_gram_matrix()
    test_get_style_cost()
