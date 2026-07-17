"""
Model definition and building.
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dense, Dropout, Concatenate, BatchNormalization, GlobalMaxPooling1D,
    RepeatVector, Permute, Multiply, Activation, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def attention_layer(inputs):
    """Custom attention layer."""
    att = Dense(1, activation='tanh')(inputs)
    att = Flatten()(att)
    att = Activation('softmax')(att)
    att = RepeatVector(inputs.shape[-1])(att)
    att = Permute([2, 1])(att)
    return Multiply()([inputs, att])

def create_model(vocab_size, seq_len, beh_dim):
    """Build the CNN‑LSTM‑Attention model."""
    text_input = Input(shape=(seq_len,), name='text_input')
    beh_input = Input(shape=(beh_dim,), name='behavioral_input')

    x = Embedding(vocab_size, 128, input_length=seq_len)(text_input)
    x = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(x)
    x_att = attention_layer(x)
    gmp = GlobalMaxPooling1D()(x_att)
    lstm_last = x[:, -1, :]
    concat = Concatenate()([gmp, lstm_last, beh_input])

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[text_input, beh_input], outputs=output)