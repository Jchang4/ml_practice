

class CONFIG:
    # Image
    HEIGHT = 512
    WIDTH = 512
    NUM_CHANNELS = 3

    # Layers for Cost Function
    CONTENT_LAYERS = ['block5_conv2']
    STYLE_LAYERS = [('block1_conv1', 0.2),
                    ('block2_conv1', 0.2),
                    ('block3_conv1', 0.2),
                    ('block4_conv1', 0.2),
                    ('block5_conv1', 0.2)]
