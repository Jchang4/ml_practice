import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19

from model import get_model
from images import load_and_process_img



if __name__ == '__main__':


    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    CONTENT_IMAGE_PATH = './data/turtle.png'
    img = load_and_process_img(CONTENT_IMAGE_PATH)
    print(img.shape)

    # m = get_model(content_layers, style_layers)
    # model = get_model(content_layers, style_layers)
    # # sess = tf.Session()
    # # output = sess.run(vgg, feed_dict={'input': img})
    # output = model.predict(img)
    #
    # print(model)
    # print(output.shape)


    model = VGG19()
    print(model.summary())
