import tensorflow as tf
from tensorflow.keras import layers, models

def create_lip_sync_model(input_shape):
    """Create a CNN model for lip sync detection."""
    model = models.Sequential([
        # Reshape the input to make it suitable for CNN
        layers.Reshape((input_shape[0] // 3, 3, 1), input_shape=(input_shape[0],)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification: speaking or not speaking
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Alternative model: LSTM for temporal features
def create_lstm_lip_sync_model(input_shape, sequence_length):
    """Create an LSTM model for lip sync detection with temporal features."""
    # Reshape input: (batch_size, sequence_length, features)
    input_layer = layers.Input(shape=(sequence_length, input_shape[0]))
    
    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(input_layer)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
