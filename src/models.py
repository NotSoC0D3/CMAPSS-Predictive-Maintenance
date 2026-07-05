import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input,
    Add, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def build_baseline_lstm(input_shape):
    """
    Builds and compiles the Baseline Stacked LSTM model (Model 1).
    """
    print("Building Baseline LSTM...")
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True, 
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bilstm(input_shape, lstm_units=[32, 16], dropout_rate=0.3, l2_weight=1e-4):
    """
    Builds the Bidirectional LSTM model (Model 2).
    """
    print("Building BiLSTM...")
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True, kernel_regularizer=regularizers.l2(l2_weight)))(inputs)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(lstm_units[1], return_sequences=False, kernel_regularizer=regularizers.l2(l2_weight)))(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_lstm_attention(input_shape):
    """
    Builds the LSTM + Multi-Head Attention model (Model 3).
    Features residual connections and global average pooling.
    """
    print("Building LSTM + Attention...")
    inputs = Input(shape=input_shape)

    # LSTM layers
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(32, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Multi-Head Attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    attn_output = Dropout(0.2)(attn_output)
    attn_output = LayerNormalization()(attn_output)

    # Add residual connection from LSTM output
    x = Add()([x, attn_output])

    # Global pooling and final projection
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_bilstm_attention(input_shape):
    """
    Builds the BiLSTM + Multi-Head Attention model (Model 4).
    The ultimate architecture combining bidirectional temporal context 
    with dynamic feature attention and residual connections.
    """
    print("Building BiLSTM + Attention...")
    inputs = Input(shape=input_shape)

    # Bidirectional LSTM Layer 1
    x = Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=l2(0.001),
                           recurrent_regularizer=l2(0.001)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Bidirectional LSTM Layer 2
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    attn_output = Dropout(0.2)(attn_output)
    attn_output = LayerNormalization()(attn_output)

    # Residual Connection
    x = Add()([x, attn_output])

    # Global Average Pooling and Output
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

