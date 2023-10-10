import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

class SP_LSTM_Cell(Layer):
    def __init__(self, units, input_dim, **kwargs):
        self.units = units
        self.input_dim = input_dim
        self.state_size = [units, units]
        super(SP_LSTM_Cell, self).__init__(**kwargs)

    def build(self, input_shape):
        num_units = self.units

        self.Wu = self.add_weight(shape=(self.input_dim + num_units, self.units), initializer='glorot_uniform')
        self.bu = self.add_weight(shape=(self.units,), initializer='zeros')

        self.Wf = self.add_weight(shape=(self.input_dim + num_units, self.units), initializer='glorot_uniform')
        self.bf = self.add_weight(shape=(self.units,), initializer='zeros')

        self.Wc = self.add_weight(shape=(self.input_dim + num_units, self.units), initializer='glorot_uniform')
        self.bc = self.add_weight(shape=(self.units,), initializer='zeros')

        self.Wo = self.add_weight(shape=(self.input_dim + num_units, self.units), initializer='glorot_uniform')
        self.bo = self.add_weight(shape=(self.units,), initializer='zeros')

        super(SP_LSTM_Cell, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, states):
        h_prev, c_prev = states

        x = tf.concat([inputs, h_prev], axis=1)

        u = tf.sigmoid(tf.matmul(x, self.Wu) + self.bu)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)
        c_tilde = tf.tanh(tf.matmul(x, self.Wc) + self.bc)
        c = u * c_tilde + f * c_prev
        o = tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)
        h = o * tf.tanh(c)

        return h, [h, c]

# SP-LSTM Layer definition
class SP_LSTM(Layer):
    def __init__(self, units, input_dim, return_sequences=False, **kwargs):
        self.units = units
        self.return_sequences = return_sequences
        self.cell = SP_LSTM_Cell(units, input_dim)
        super(SP_LSTM, self).__init__(**kwargs)

    def call(self, inputs):
        h_prev = tf.zeros((tf.shape(inputs)[0], self.units))
        c_prev = tf.zeros((tf.shape(inputs)[0], self.units))
        outputs = []

        for t in range(inputs.shape[1]):
            output, [h_prev, c_prev] = self.cell(inputs[:, t, :], [h_prev, c_prev])
            outputs.append(output)

        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return outputs[-1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'input_dim': self.input_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        cell_config = config.pop('cell')
        cell = getattr(sys.modules[__name__], cell_config.pop('class_name')).from_config(cell_config['config'])
        return cls(cell=cell, **config)

