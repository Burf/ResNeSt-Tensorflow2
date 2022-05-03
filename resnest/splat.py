import tensorflow as tf

def group_conv(x, filters = None, kernel_size = 3, **kwargs):
    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size]
    n_group = len(kernel_size)
    if n_group == 1:
        out = [x]
    else:
        size = tf.keras.backend.int_shape(x)[-1]
        split_size = [size // n_group if index != 0 else size // n_group + size % n_group for index in range(n_group)]
        out = tf.split(x, split_size, axis = -1)
    
    name = None
    if "name" in kwargs:
        name = kwargs["name"]
    result = []
    for index in range(n_group):
        kwargs["filters"] = filters // n_group
        if index == 0:
            kwargs["filters"] += filters % n_group
        kwargs["kernel_size"] = kernel_size[index]
        if name is not None and 1 < n_group:
            kwargs["name"] = "{0}_group{1}".format(name, index + 1)
        result.append(tf.keras.layers.Conv2D(**kwargs)(out[index]))
    if n_group == 1:
        out = result[0]
    else:
        out = tf.keras.layers.Concatenate(axis = -1, name = name)(result)
    return out

def split_attention_block(x, n_filter, kernel_size = 3, stride_size = 1, dilation = 1, group_size = 1, radix = 1, dropout_rate = 0., expansion = 4, prefix = ""):
    if len(prefix) != 0:
        prefix += "_"
    out = group_conv(x, n_filter * radix, [kernel_size] * (group_size * radix), strides = stride_size, dilation_rate = dilation, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "{0}split_attention_conv1".format(prefix))
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "{0}split_attention_bn1".format(prefix))(out)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "{0}split_attention_dropout1".format(prefix))(out)
    out = tf.keras.layers.Activation(tf.keras.activations.relu, name = "{0}split_attention_act1".format(prefix))(out)
    
    inter_channel = max(tf.keras.backend.int_shape(x)[-1] * radix // expansion, 32)
    if 1 < radix:
        split = tf.split(out, radix, axis = -1)
        out = tf.keras.layers.Add(name = "{0}split_attention_add".format(prefix))(split)
    out = tf.keras.layers.GlobalAveragePooling2D(name = "{0}split_attention_gap".format(prefix))(out)
    out = tf.keras.layers.Reshape([1, 1, n_filter], name = "{0}split_attention_expand_dims".format(prefix))(out)
    
    out = group_conv(out, inter_channel, [1] * group_size, padding = "same", use_bias = True, kernel_initializer = "he_normal", name = "{0}split_attention_conv2".format(prefix))
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "{0}split_attention_bn2".format(prefix))(out)
    out = tf.keras.layers.Activation("relu", name = "{0}split_attention_act2".format(prefix))(out)
    out = group_conv(out, n_filter * radix, [1] * group_size, padding = "same", use_bias = True, kernel_initializer = "he_normal", name = "{0}split_attention_conv3".format(prefix))
    
    #attention = rsoftmax(out, n_filter, radix, group_size)
    attention = rSoftMax(n_filter, radix, group_size, name = "{0}split_attention_softmax".format(prefix))(out)
    if 1 < radix:
        attention = tf.split(attention, radix, axis = -1)
        out = tf.keras.layers.Add(name = "{0}split_attention_out".format(prefix))([o * a for o, a in zip(split, attention)])
    else:
        out = tf.keras.layers.Multiply(name = "{0}split_attention_out".format(prefix))([attention, out])
    return out
    
def rsoftmax(x, n_filter, radix, group_size):
    if 1 < radix:
        out = tf.keras.layers.Reshape([group_size, radix, n_filter // group_size])(x)
        out = tf.keras.layers.Permute([2, 1, 3])(out)
        out = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis = 1))(out)
        out = tf.keras.layers.Reshape([1, 1, radix * n_filter])(out)
    else:
        out = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)
    return out
    
class rSoftMax(tf.keras.layers.Layer):
    def __init__(self, filters, radix, group_size, **kwargs):
        super(rSoftMax, self).__init__(**kwargs)
        
        self.filters = filters
        self.radix = radix
        self.group_size = group_size
        
        if 1 < radix:
            self.seq1 = tf.keras.layers.Reshape([group_size, radix, filters // group_size])
            self.seq2 = tf.keras.layers.Permute([2, 1, 3])
            self.seq3 = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis = 1))
            self.seq4 = tf.keras.layers.Reshape([1, 1, radix * filters])
            self.seq = [self.seq1, self.seq2, self.seq3, self.seq4]
        else:
            self.seq1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            self.seq = [self.seq1]

    def call(self, inputs):
        out = inputs
        for l in self.seq:
            out = l(out)
        return out
    
    def get_config(self):
        config = super(rSoftMax, self).get_config()
        config["filters"] = self.filters
        config["radix"] = self.radix
        config["group_size"] = self.group_size
        return config
