from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation

def build_model(image_size,
                n_classes,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=[0.5, 1.0, 2.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                limit_boxes=True,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='centroids',
                normalize_coords=False):

    n_predictor_layers = 4

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratio_global` and 'aspect_ratios_per_layer` cannot both be None, At leaset one needs to be specified")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, \
                             but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))
    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, \
                             but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be > 0, but the variances given are {}".format(variances))

    if aspect_ratios_per_layer:
        aspect_ratios_conv4 = aspect_ratios_per_layer[0]
        aspect_ratios_conv5 = aspect_ratios_per_layer[1]
        aspect_ratios_conv6 = aspect_ratios_per_layer[2]
        aspect_ratios_conv7 = aspect_ratios_per_layer[3]
    else:
        aspect_ratios_conv4 = aspect_ratios_global
        aspect_ratios_conv5 = aspect_ratios_global
        aspect_ratios_conv6 = aspect_ratios_global
        aspect_ratios_conv7 = aspect_ratios_global

    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1)
            else:
                n_boxes.append(len(aspect_ratios))
        n_boxes_conv4 = n_boxes[0]
        n_boxes_conv5 = n_boxes[1]
        n_boxes_conv6 = n_boxes[2]
        n_boxes_conv7 = n_boxes[3]
    else:
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes_conv4 = n_boxes
        n_boxes_conv5 = n_boxes
        n_boxes_conv6 = n_boxes
        n_boxes_conv7 = n_boxes

    # Base Network
    x = Input(shape=image_size)
    normed = Lambda(lambda z:z/127.5 - 1.,
                    output_shape=image_size, name='input_norm')(x)

    conv1 = Conv2D(32, (5, 5), name='conv1', strides=(1, 1), padding='same')(normed)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (5, 5), name='conv2', strides=(1, 1), padding='same')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (5, 5), name='conv3', strides=(1, 1), padding='same')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(64, (5, 5), name='conv4', strides=(1, 1), padding='same')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (5, 5), name='conv5', strides=(1, 1), padding='same')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (5, 5), name='conv6', strides=(1, 1), padding='same')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (5, 5), name='conv7', strides=(1, 1), padding='same')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)