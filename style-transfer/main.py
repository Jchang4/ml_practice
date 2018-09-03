import tensorflow as tf
from keras.models import Model

def get_model(style_layers, content_layers):
    """
    Take an input image and output content + style layers
    """
    # Load VGG19
    vgg = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
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


if __name__ == '__main__':
    tf.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    sess = tf.ImmediateSession()
    print(sess.run(tf.add([1], [1])))
