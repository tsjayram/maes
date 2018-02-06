from keras.layers import Input, Dense, Reshape, Concatenate
from keras.models import Model
from keras.activations import sigmoid, linear


class Controller():
    def __init__(self, tm_output_units, output_trainable,
                 tm_in_dim, tm_state_units, n_heads, M):
        self.heads_flat_dim = n_heads * M
        tm_ctrl_in_dim = tm_in_dim + tm_state_units + self.heads_flat_dim
        tm_ctrl_inputs = Input(shape=(tm_ctrl_in_dim,))
        tm_hidden = Dense(tm_ctrl_in_dim, activation=linear, trainable=output_trainable,
                          name='Controller_hidden')(tm_ctrl_inputs)
        tm_output = Dense(tm_output_units, activation=sigmoid, trainable=output_trainable,
                          name='Controller_out')(tm_hidden)

        tm_state = Dense(tm_state_units, name='Controller_state')(tm_ctrl_inputs)

        self.controller = Model(tm_ctrl_inputs, [tm_output, tm_state])

        self.trainable_weights = self.controller.trainable_weights

    def call(self, tm_input, tm_state, data_heads):
        heads_flat = Reshape((self.heads_flat_dim,))(data_heads)
        tm_ctrl_inputs = Concatenate()([tm_input, tm_state, heads_flat])
        tm_output, tm_state = self.controller(tm_ctrl_inputs)
        return tm_output, tm_state
