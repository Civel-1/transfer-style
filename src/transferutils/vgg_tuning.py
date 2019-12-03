from typing import Sequence

from tensorflow import keras
from tensorflow.keras.models import Model


def replace_max_by_average_pooling(model: Model) -> Model:

    input_layer, *other_layers = model.layers
    assert isinstance(input_layer, keras.layers.InputLayer)

    x = input_layer.output
    for layer in other_layers:
        if isinstance(layer, keras.layers.MaxPooling2D):
            layer = keras.layers.AveragePooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                name=layer.name,
            )
        x = layer(x)

    return Model(inputs=input_layer.input, outputs=x)


def get_vgg_layer(model: Model, layer_name: str, model_name: str=None) -> keras.models.Model:
    layer = model.get_layer(layer_name)
    try:
        output = layer.get_output_at(1)
    except:
        output = layer.get_output_at(0)
    return keras.models.Model(model.layers[0].input, output, name=model_name)