from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .mobilenetv2 import MobileNetV2 
from .resnetv2 import ResNet18, ResNet50, ResNet101
from .vgg  import vgg19_bn, vgg16_bn, vgg13_bn, vgg13_bn_3neurons, vgg11_bn, vgg8_bn, vgg8_bn_3neurons
from .wrn import wrn_40_2, wrn_16_2
model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'mobilenetv2': MobileNetV2,
    'resnet18' : ResNet18 ,
    'resnet50' : ResNet50,
    'vgg8': vgg8_bn,
    'vgg8_3neurons': vgg8_bn_3neurons,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg13_3neurons': vgg13_bn_3neurons,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'wrn_40_2' : wrn_40_2,
    'wrn_16_2' : wrn_16_2 
}
