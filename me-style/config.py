
# Image Size
TARGET_HEIGHT = 512
TARGET_WIDTH = 512
# TARGET_HEIGHT = 224
# TARGET_WIDTH = 224
TARGET_CHANNELS = 3


# Layers for Cost Function
CONTENT_LAYER = 'block5_conv2'
STYLE_LAYERS = [('block1_conv1', 0.2),
                ('block2_conv1', 0.2),
                ('block3_conv1', 0.2),
                ('block4_conv1', 0.2),
                ('block5_conv1', 0.2)]
