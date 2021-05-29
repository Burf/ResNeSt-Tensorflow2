import tensorflow as tf
import torch

from .resnet import ResNet

#_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'
_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def load_weight(keras_model, torch_url, group_size = 2):
    """
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest50-528c19ca.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest200-75117900.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest200-75117900.pth
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest269-0cc87c48.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest269-0cc87c48.pth
    """
    torch_weight = torch.hub.load_state_dict_from_url(torch_url, progress = True, check_hash = True)
    
    weight = {}
    for k, v in dict(torch_weight).items():
        if k.split(".")[-1] in ["weight", "bias", "running_mean", "running_var"]:
            if ("downsample" in k or "conv" in k) and "weight" in k and v.ndim == 4:
                v = v.permute(2, 3, 1, 0)
            elif "fc.weight" in k:
                v = v.t()
            weight[k] = v.data.numpy()
    
    g = 0
    downsample = []
    keras_weight = []
    for i, (torch_name, torch_weight) in enumerate(weight.items()):
        if i + g < len(keras_model.weights):
            keras_name = keras_model.weights[i + g].name
            if "downsample" in torch_name:
                downsample.append(torch_weight)
                continue
            elif "group" in keras_name:
                g += (group_size - 1)
                torch_weight = tf.split(torch_weight, group_size, axis = -1)
            else:
                torch_weight = [torch_weight]
            keras_weight += torch_weight
    
    for w in keras_model.weights:
        if "downsample" in w.name:
            new_w = downsample.pop(0)
        else:
            new_w = keras_weight.pop(0)
        tf.keras.backend.set_value(w, new_w)
    return keras_model
    
def resnest50(include_top = True, weights = "imagenet", input_tensor = None, input_shape = None, classes = 1000):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    
    out = ResNet(img_input, [3, 4, 6, 3], classes, include_top, radix = 2, group_size = 1, block_width = 64, stem_width = 32, deep_stem = True, avg_down = True, avd = True, avd_first = False)
    model = tf.keras.Model(img_input, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest50"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
    return model
    

def resnest101(include_top = True, weights = "imagenet", input_tensor = None, input_shape = None, classes = 1000):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    
    out = ResNet(img_input, [3, 4, 23, 3], classes, include_top, radix = 2, group_size = 1, block_width = 64, stem_width = 64, deep_stem = True, avg_down = True, avd = True, avd_first = False)
    model = tf.keras.Model(img_input, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest101"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
    return model

def resnest200(include_top = True, weights = "imagenet", input_tensor = None, input_shape = None, classes = 1000):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
            
    out = ResNet(img_input, [3, 24, 36, 3], classes, include_top, radix = 2, group_size = 1, block_width = 64, stem_width = 64, deep_stem = True, avg_down = True, avd = True, avd_first = False)
    model = tf.keras.Model(img_input, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest200"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
    return model

def resnest269(include_top = True, weights = "imagenet", input_tensor = None, input_shape = None, classes = 1000):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
            
    out = ResNet(img_input, [3, 30, 48, 8], classes, include_top, radix = 2, group_size = 1, block_width = 64, stem_width = 64, deep_stem = True, avg_down = True, avd = True, avd_first = False)
    model = tf.keras.Model(img_input, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest269"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
    return model
