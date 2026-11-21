"""
CBAM: Convolutional Block Attention Module (Woo et al., 2018)

Two sequential sub-modules applied to a feature map:
1. Channel attention: "what" to attend to -- learns which feature channels
   matter most, using both average-pooled and max-pooled channel
   descriptors through a shared MLP.
2. Spatial attention: "where" to attend to -- learns which spatial
   locations matter most, using channel-pooled (avg + max) feature maps
   through a small conv layer.

Applied in sequence: input -> channel attention -> spatial attention -> output.
This matches the original CBAM paper's design and is the standard placement
used in most CBAM + EfficientNet papers.
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def channel_attention(input_tensor, reduction_ratio: int = 16, name: str = "channel_attn"):
    channels = input_tensor.shape[-1]

    shared_dense_1 = layers.Dense(channels // reduction_ratio, activation="relu",
                                   name=f"{name}_dense1")
    shared_dense_2 = layers.Dense(channels, name=f"{name}_dense2")

    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_avgpool")(input_tensor)
    avg_out = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_maxpool")(input_tensor)
    max_out = shared_dense_2(shared_dense_1(max_pool))

    attention = layers.Add(name=f"{name}_add")([avg_out, max_out])
    attention = layers.Activation("sigmoid", name=f"{name}_sigmoid")(attention)
    attention = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(attention)

    return layers.Multiply(name=f"{name}_scale")([input_tensor, attention])


def spatial_attention(input_tensor, kernel_size: int = 7, name: str = "spatial_attn"):
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), name=f"{name}_avgpool"
    )(input_tensor)
    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True), name=f"{name}_maxpool"
    )(input_tensor)

    concat = layers.Concatenate(axis=-1, name=f"{name}_concat")([avg_pool, max_pool])
    attention = layers.Conv2D(
        filters=1, kernel_size=kernel_size, padding="same",
        activation="sigmoid", name=f"{name}_conv"
    )(concat)

    return layers.Multiply(name=f"{name}_scale")([input_tensor, attention])


def cbam_block(input_tensor, reduction_ratio: int = 16, kernel_size: int = 7, name: str = "cbam"):
    """
    Full CBAM block: channel attention followed by spatial attention.

    Placement (per the locked architecture spec): apply this to the
    backbone's final conv feature map, BEFORE global average pooling.
    For EfficientNetB0 at 224x224 input, this feature map is 7x7x1280.
    """
    x = channel_attention(input_tensor, reduction_ratio, name=f"{name}_channel")
    x = spatial_attention(x, kernel_size, name=f"{name}_spatial")
    return x