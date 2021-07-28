import tensorflow as tf
from .splat import *

def resnest_block(x, n_filter, stride_size = 1, dilation = 1, group_size = 1, radix = 1, block_width = 64, avd = False, avd_first = False, downsample = None, dropout_rate = 0., expansion = 4, is_first = False, stage = 1, index = 1):
    avd = avd and (1 < stride_size or is_first)
    group_width = int(n_filter * (block_width / 64)) * group_size
    
    out = tf.keras.layers.Conv2D(group_width, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_block{1}_conv1".format(stage, index))(x)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_block{1}_bn1".format(stage, index))(out)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "stage{0}_block{1}_dropout1".format(stage, index))(out)
    out = tf.keras.layers.Activation("relu", name = "stage{0}_block{1}_act1".format(stage, index))(out)
    
    if avd:
        avd_layer = tf.keras.layers.AveragePooling2D(3, strides = stride_size, padding = "same", name = "stage{0}_block{1}_avd".format(stage, index))
        stride_size = 1
        if avd_first:
            out = avd_layer(out)

    if 0 < radix:
        out = split_attention_block(out, group_width, 3, stride_size, dilation, group_size, radix, dropout_rate, expansion, prefix = "stage{0}_block{1}".format(stage, index))
    else:
        out = tf.keras.layers.Conv2D(group_width, 3, strides = stride_size, dilation_rate = dilation, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_block{1}_conv2".format(stage, index))(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_block{1}_bn2".format(stage, index))(out)
        if 0 < dropout_rate:
            out = tf.keras.layers.Dropout(dropout_rate, name = "stage{0}_block{1}_dropout2".format(stage, index))(out)
        out = tf.keras.layers.Activation("relu", name = "stage{0}_block{1}_act2".format(stage, index))(out)
    
    if avd and not avd_first:
        out = avd_layer(out)
    
    out = tf.keras.layers.Conv2D(n_filter * expansion, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_block{1}_conv3".format(stage, index))(out)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_block{1}_bn3".format(stage, index))(out)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "stage{0}_block{1}_dropout3".format(stage, index))(out)
    residual = x
    if downsample is not None:
        residual = downsample
    out = tf.keras.layers.Add(name = "stage{0}_block{1}_shorcut".format(stage, index))([out, residual])
    out = tf.keras.layers.Activation(tf.keras.activations.relu, name = "stage{0}_block{1}_shorcut_act".format(stage, index))(out)
    return out

def resnest_module(x, n_filter, n_block, stride_size = 1, dilation = 1, group_size = 1, radix = 1, block_width = 64, avg_down = True, avd = False, avd_first = False, dropout_rate = 0., expansion = 4, is_first = True, stage = 1):
    downsample = None
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != (n_filter * expansion):
        if avg_down:
            if dilation == 1:
                downsample = tf.keras.layers.AveragePooling2D(stride_size, strides = stride_size, padding = "same", name = "stage{0}_downsample_avgpool".format(stage))(x)
            else:
                downsample = tf.keras.layers.AveragePooling2D(1, strides = 1, padding = "same", name = "stage{0}_downsample_avgpool".format(stage))(x)
            downsample = tf.keras.layers.Conv2D(n_filter * expansion, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_downsample_conv1".format(stage))(downsample)
            downsample = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_downsample_bn1".format(stage))(downsample)
        else:
            downsample = tf.keras.layers.Conv2D(n_filter * expansion, 1, strides = stride_size, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_downsample_conv1".format(stage))(x)
            downsample = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_downsample_bn1".format(stage))(downsample)
    
    if dilation == 1 or dilation == 2 or dilation == 4:
        out = resnest_block(x, n_filter, stride_size, 2 ** (dilation // 4), group_size, radix, block_width, avd, avd_first, downsample, dropout_rate, expansion, is_first, stage = stage)
    else:
        raise ValueError("unknown dilation size '{0}'".format(dilation))
    
    for index in range(1, n_block):
        out = resnest_block(out, n_filter, 1, dilation, group_size, radix, block_width, avd, avd_first, dropout_rate = dropout_rate, expansion = expansion, stage = stage, index = index + 1)
    return out

def ResNet(x, stack, n_class = 1000, include_top = True, dilation = 1, group_size =1, radix = 1, block_width = 64, stem_width = 64, deep_stem = False, dilated = False, avg_down = False, avd = False, avd_first = False, dropout_rate = 0., expansion = 4):
    #Stem
    if deep_stem:
        out = tf.keras.layers.Conv2D(stem_width, 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv1")(x)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn1")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act1")(out)
        out = tf.keras.layers.Conv2D(stem_width, 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv2")(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn2")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act2")(out)
        out = tf.keras.layers.Conv2D(stem_width * 2, 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv3")(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn3")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act3")(out)
    else:
        out = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv1")(x)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn1")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act1")(out)
    out = tf.keras.layers.MaxPool2D(3, strides = 2, padding = "same", name = "stem_pooling")(out)
    
    #Stage 1
    out = resnest_module(out, 64, stack[0], 1, 1, group_size, radix, block_width, avg_down, avd, avd_first, expansion = expansion, is_first = False, stage = 1)
    #Stage 2
    out = resnest_module(out, 128, stack[1], 2, 1, group_size, radix, block_width, avg_down, avd, avd_first, expansion = expansion, stage = 2)
    
    if dilated or dilation == 4:
        dilation = [2, 4]
        stride_size = [1, 1]
    elif dilation == 2:
        dilation = [1, 2]
        stride_size = [2, 1]
    else:
        dilation = [1, 1]
        stride_size = [2, 2]
    
    #Stage 3
    out = resnest_module(out, 256, stack[2], stride_size[0], dilation[0], group_size, radix, block_width, avg_down, avd, avd_first, dropout_rate, expansion, stage = 3)
    #Stage 4
    out = resnest_module(out, 512, stack[3], stride_size[1],dilation[1], group_size, radix, block_width, avg_down, avd, avd_first, dropout_rate, expansion, stage = 4)
    
    if include_top:
        out = tf.keras.layers.GlobalAveragePooling2D(name = "feature_avg_pool")(out)
        out = tf.keras.layers.Dense(n_class, activation = tf.keras.activations.softmax, name = "logits")(out)
    return out
