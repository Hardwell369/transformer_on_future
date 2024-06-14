import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Layer, LayerNormalization, Dropout, GlobalaveragePooling1D
from tensorflow.keras.models import Model
import keras.saving

# 定义位置编码层
@keras.utils.register_keras_serializable(package='Custom', name='PositionalEncoding')
class PositionalEncoding(Layer):
    def __init__(self, position, d_model, dtype=tf.float32, name="PositionalEncoding", trainable=True):
        super(PositionalEncoding, self).__init__(dtype=dtype, name=name, trainable=trainable)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        pos = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rads = self.get_angles(pos, i, d_model)

        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="tanh")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    inputs = PositionalEncoding(inputs.shape[1], inputs.shape[2])(inputs)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(mlp_units, activation="tanh")(x)
    x = Dropout(mlp_dropout)(x)
    x = GlobalaveragePooling1D(data_format="channels_last")(x)
    x = Dense(1)(x)
    return Model(inputs, x)
