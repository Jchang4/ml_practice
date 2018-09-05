from keras.models import Model
from keras.applications.vgg19 import VGG19


# VGG19
def get_vgg_model():
    """
    Return VGG19 Model
    """
    vgg = VGG19(include_top=True, weights='imagenet')
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.output)

def get_features_from_model(model, layer_name):
    return model.get_layer(layer_name).output
