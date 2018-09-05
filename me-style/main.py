
from helpers.image import get_image, vgg_preprocess
from helpers.model import get_vgg_model, get_features_from_model


content_path = './data/turtle.png'
style_path = './data/van-gogh.jpg'

content_image = get_image(content_path, show=True)
style_image = get_image(style_path, show=True)
