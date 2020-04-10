from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet
from configuration import model_index

def get_model():
    if model_index == 0:
        return mobilenet_v1.MobileNetV1()
    elif model_index == 1:
        return mobilenet_v2.MobileNetV2()
    elif model_index == 2:
        return mobilenet_v3_large.MobileNetV3Large()
    elif model_index == 3:
        return mobilenet_v3_small.MobileNetV3Small()
    elif model_index == 4:
        return efficientnet.efficient_net_b0()
    elif model_index == 5:
        return efficientnet.efficient_net_b1()
    elif model_index == 6:
        return efficientnet.efficient_net_b2()
    elif model_index == 7:
        return efficientnet.efficient_net_b3()
    elif model_index == 8:
        return efficientnet.efficient_net_b4()
    elif model_index == 9:
        return efficientnet.efficient_net_b5()
    elif model_index == 10:
        return efficientnet.efficient_net_b6()
    elif model_index == 11:
        return efficientnet.efficient_net_b7()
    elif model_index == 12:
        return resnext.ResNeXt50()
    elif model_index == 13:
        return resnext.ResNeXt101()
    elif model_index == 14:
        return inception_v4.InceptionV4()
    elif model_index == 15:
        return inception_resnet_v1.InceptionResNetV1()
    elif model_index == 16:
        return inception_resnet_v2.InceptionResNetV2()
    elif model_index == 17:
        return se_resnet.se_resnet_50()
    elif model_index == 18:
        return se_resnet.se_resnet_101()
    elif model_index == 19:
        return se_resnet.se_resnet_152()
    elif model_index == 20:
        return squeezenet.SqueezeNet()
    elif model_index == 21:
        return densenet.densenet_121()
    elif model_index == 22:
        return densenet.densenet_169()
    elif model_index == 23:
        return densenet.densenet_201()
    elif model_index == 24:
        return densenet.densenet_264()
    elif model_index == 25:
        return shufflenet_v2.shufflenet_0_5x()
    elif model_index == 26:
        return shufflenet_v2.shufflenet_1_0x()
    elif model_index == 27:
        return shufflenet_v2.shufflenet_1_5x()
    elif model_index == 28:
        return shufflenet_v2.shufflenet_2_0x()
    elif model_index == 29:
        return resnet.resnet_18()
    elif model_index == 30:
        return resnet.resnet_34()
    elif model_index == 31:
        return resnet.resnet_50()
    elif model_index == 32:
        return resnet.resnet_101()
    elif model_index == 33:
        return resnet.resnet_152()
    else:
        raise ValueError("The model_index does not exist.")