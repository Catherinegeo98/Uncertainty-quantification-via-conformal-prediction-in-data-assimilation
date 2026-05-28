from keras.layers import Input, Conv1D, Lambda
from keras.models import Model
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
def apply_pos_r(z):
    return tf.concat(
        [z[..., 0:1], z[..., 1:2], tf.nn.softplus(z[..., 2:3])],
        axis=-1)


def fully_convolutional_two_heads(filters, kernels, activation='selu',
                                   positive_r=True, inn=4):
    insize = sum(kernels) - len(kernels) + 250
    inp = Input(shape=(insize, inn))

    x = Conv1D(filters[0], kernel_size=kernels[0], padding='valid',
               activation=activation)(inp)
    for i in range(1, len(filters)):
        x = Conv1D(filters[i], kernel_size=kernels[i], padding='valid',
                   activation=activation)(x)

    # Lower quantile 
    lower_raw = Conv1D(3, kernel_size=1, padding='same')(x)
    if positive_r:
        lower = Lambda(apply_pos_r, output_shape=lambda s: s, name='lower')(lower_raw)
    else:
        lower = Lambda(lambda z: z, output_shape=lambda s: s, name='lower')(lower_raw)

    # Upper quantile
    upper_raw = Conv1D(3, kernel_size=1, padding='same')(x)
    if positive_r:
        upper = Lambda(apply_pos_r, output_shape=lambda s: s, name='upper')(upper_raw)
    else:
        upper = Lambda(lambda z: z, output_shape=lambda s: s, name='upper')(upper_raw)

    return Model(inp, [lower, upper])
 
 