# This code has been inspired by the code provided with the Maestro tool: https://github.com/maestro-project/maestro/blob/master/tools/frontend/frameworks_to_modelfile_maestro.py

import re
import os
import argparse
from argparse import RawTextHelpFormatter

import torch
import torchvision.models as models

import tensorflow as tf
import tensorflow.keras.applications as keras_models
import yaml


def get_model(model_name, input_shape):
    keras_model_name = ''
    if(model_name == 'xception'):
        keras_model_name = 'Xception'

    elif(model_name == 'vgg16'):
        keras_model_name = 'VGG16'
    elif(model_name == 'vgg19'):
        keras_model_name = 'VGG19'

    elif(model_name == 'resnet50'):
        keras_model_name = 'ResNet50'
    elif(model_name == 'resnet101'):
        keras_model_name = 'ResNet101'
    elif(model_name == 'resnet152'):
        keras_model_name = 'ResNet152'
    elif(model_name == 'resnet50_v2'):
        keras_model_name = 'ResNet50V2'
    elif(model_name == 'resnet101_v2'):
        keras_model_name = 'ResNet101V2'
    elif(model_name == 'resnet152_v2'):
        keras_model_name = 'ResNet152V2'

    elif(model_name == 'inception_v3'):
        keras_model_name = 'InceptionV3'
    elif(model_name == 'inception_resnet_v2'):
        keras_model_name = 'InceptionResNetV2'

    elif(model_name == 'mobilenet'):
        keras_model_name = 'MobileNet'
    elif(model_name == 'mobilenet_v2'):
        keras_model_name = 'MobileNetV2'

    elif(model_name == 'densenet121'):
        keras_model_name = 'DenseNet121'
    elif(model_name == 'densenet169'):
        keras_model_name = 'DenseNet169'
    elif(model_name == 'densenet201'):
        keras_model_name = 'DenseNet201'

    elif(model_name == 'nasnet_large'):
        keras_model_name = 'NASNetLarge'
    elif(model_name == 'nasnet_mobile'):
        keras_model_name = 'NASNetMobile'

    else:
        raise NotImplementedError('Not supported model')
    keras_model = getattr(keras_models, keras_model_name)(weights=None, include_top=True, input_shape=input_shape)
    print('Get the keras model: ' + keras_model_name)
    return keras_model

# Retrieve the layer output shape for models
def get_output_size(W, H, kernel_size, stride, padding):
    W_out = int((W - kernel_size + 2 * padding) / stride) + 1
    H_out = int((H - kernel_size + 2 * padding) / stride) + 1
    # dilation = 1
    # W_out = int((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    # H_out = int((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return W_out, H_out

# Get layer dimensions from pytorch model summary
def get_conv_layers(model, input_size, batch_size):
    layers = []
    W, H, C = input_size
    N = batch_size
    print(f"Number of layers in {opt.model}: {len(list(model.modules()))}")
    print([print(f"{i} layer name: {name}, layer type: {type(layer)}") for i, (name, layer) in enumerate(list(model.named_modules()))])
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            M = m.out_channels
            S = m.kernel_size[0]
            R = m.kernel_size[1]
            Wpad = m.padding[0]
            Hpad = m.padding[1]
            Wstride = m.stride[0]
            Hstride = m.stride[1]

            layer = (W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer)

            #print(f"in_size: {W}x{H}")
            W, H = get_output_size(W, H, S, Wstride, Wpad)
            #print(f"out_size: {W}x{H}")
            C = M
        
        if isinstance(m, torch.nn.MaxPool2d):
            Wstride = m.stride
            Hstride = m.stride
            W = W // Wstride
            H = H // Hstride
    return layers



# Get layer dimensions from pytorch model summary
def get_conv_layers_keras(model, input_size, batch_size):
    layers = []
    W, H, C = input_size
    N = batch_size

    for m in model.layers:
        if isinstance(m, tf.keras.layers.Conv2D) or isinstance(m, tf.keras.layers.DepthwiseConv2D):
            if isinstance(m, tf.keras.layers.DepthwiseConv2D):
                M = C
            else:
                M = m.filters

            S = m.kernel_size[0]
            R = m.kernel_size[1]
            Wpad = m.padding
            Hpad = m.padding

            if Wpad == 'same':
                Wpad = (S - 1) // 2
                Hpad = (R - 1) // 2
            elif Wpad == 'valid':
                Wpad = 0
                Hpad = 0

            Wstride = m.strides[0]
            Hstride = m.strides[1]

            layer = (W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer)

            W, H = get_output_size(W, H, S, Wstride, Wpad)
            C = M

        if isinstance(m, tf.keras.layers.MaxPooling2D):
            Wstride = m.strides[0]
            Hstride = m.strides[1]
            W = W // Wstride
            H = H // Hstride
    return layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, description="Parser of keras/pytorch models into Timeloop layer description format")
    parser.add_argument('-a', '--api_name', type=str, default="pytorch", help="api choices: pytorch, keras")
    parser.add_argument('-i', '--input_size', type=str, default="224,224,3", help='input size in format W,H,C')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-m', '--model', type=str, default="mobilenet_v2", help='model from torchvision choices: \n'
                                                                        'resnet18, alexnet, vgg16, squeezenet, densenet, \n'
                                                                        'inception_v3, googlenet, shufflenet, \n'
                                                                        'mobilenet_v2, wide_resnet50_2, mnasnet,\n'
                                                                        '-----\n'
                                                                        'model from tensorflow.keras.applications choices: \n'
                                                                        'xception, vgg16, vgg19, resnet50, resnet101, \n'
                                                                        'resnet152, resnet50_v2, resnet101_v2, resnet152_v2, \n'
                                                                        'inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2,\n'
                                                                        'densenet121, densenet169, densenet201, nasnet_large, \n'
                                                                        'nasnet_mobile\n'
                                                                        '-----\n')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-o', '--outfile', type=str, default=f"model_layers", help='output file name')

    # Parse arguments   
    opt = parser.parse_args()
    input_size = tuple((int(d) for d in str.split(opt.input_size, ",")))

    if opt.verbose:
        print('Begin processing')
        print('API name: ' + str(opt.api_name))
        print('Model name: ' + str(opt.model))
        print('Input size: ' + str(input_size))
    
    out_dir = "parsed_models"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process model based on chosen API and return layer dimensions
    if opt.api_name == 'pytorch':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = getattr(models, opt.model)()
        model = model.to(device)

        cnn_layers = get_conv_layers(model, input_size, opt.batch_size)
        
        if opt.verbose:
            print("# Model: " + str(opt.model))
            print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
            print("cnn_layers = [")
            for layer in cnn_layers:
                print("    " + str(layer) + ",")
            print("]")
        
        with open(os.path.join(out_dir, opt.outfile + ".yaml"), "w") as f:
            f.write("api: " + str(opt.api_name) + "\n")
            f.write("model: " + str(opt.model) + "\n")
            f.write("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride\n")
            f.write("cnn_layers: \n")
            for layer in cnn_layers:
                f.write("  - [" + str(layer) + ",\n")
            f.write("]\n")

    elif opt.api_name == 'keras':    
        model = get_model(opt.model, input_size)

        cnn_layers = get_conv_layers_keras(model, input_size, opt.batch_size)

        if opt.verbose:
            print("# Model: " + str(opt.model))
            print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
            print("cnn_layers = [")
            for layer in cnn_layers:
                print("    " + str(layer) + ",")
            print("]")
            
        with open(os.path.join(out_dir, opt.outfile + ".yaml"), "w") as f:
            f.write(f"api: {opt.api_name}\n")
            f.write(f"model: {opt.model}\n")
            f.write("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride\n")
            f.write("layers:\n")
            for layer in cnn_layers:
                f.write("  - [")
                f.write(", ".join(str(p) for p in layer))
                f.write("]\n")
