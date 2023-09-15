import tensorflow as tf
from tensorflow.keras import layers as lay
import torch as tc
from torch import nn
import os
APP_DIR = "."
# import sys

MODEL_PATH = f"{APP_DIR}/models/.h5"

# tf_model = tf.keras.models.load_model()

def activation_convert(tf_layer):
    act = tf_layer.activation.__name__.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "softmax":
        dims = len(tf_layer.shape)
        return nn.Softmax(dim=dims)
    elif act == "tanh":
        return nn.Tanh()
    else:
        return None

class TorchModel(nn.Module):
    def __init__(self, tf_model):
        super().__init__()
        self.layers = []

        for layer in tf_model.layers:
            if isinstance(layer, lay.Dense):
                in_features = layer.input_shape[-1]
                out_features = layer.units
                lin_layer = nn.Linear(in_features, out_features)
                lin_layer.weight.data = tc.from_numpy(layer.get_weights()[0].T)
                lin_layer.bias.data = tc.from_numpy(layer.get_weights()[1])
                self.layers.append(lin_layer)
                actviation = activation_convert(layer)
                if not actviation == None:
                    self.layers.append(actviation)
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)


