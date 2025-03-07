import numpy as np
import tensorflow as tf
from model import create_lip_sync_model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the lip sync detection model."""
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_plots", exist_ok=True)
    
    # Create model
    input_shape = (X_train.shape[1],)
    model = create_lip_sync_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        "models/lip_sync_model_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save the final model
    model.save("models/lip_sync_model_final.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig("training_plots/training_history.png")
    plt.close()
    
    return model, history

# Example usage
if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("processed_data/X_train.npy")
    y_train = np.load("processed_data/y_train.npy")
    X_val = np.load("processed_data/X_val.npy")
    y_val = np.load("processed_data/y_val.npy")
    
    model, history = train_model(X_train, y_train, X_val, y_val)
