import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_dir):
    """Load the dataset from the processed files."""
    features = []
    labels = []
    
    # Load speaking examples
    speaking_dir = os.path.join(dataset_dir, "speaking")
    for file in os.listdir(speaking_dir):
        if file.endswith(".npy"):
            feature = np.load(os.path.join(speaking_dir, file))
            features.append(feature)
            labels.append(1)  # 1 for speaking
    
    # Load not speaking examples
    not_speaking_dir = os.path.join(dataset_dir, "not_speaking")
    for file in os.listdir(not_speaking_dir):
        if file.endswith(".npy"):
            feature = np.load(os.path.join(not_speaking_dir, file))
            features.append(feature)
            labels.append(0)  # 0 for not speaking
    
    return np.array(features), np.array(labels)

def prepare_dataset(dataset_dir, test_size=0.2, validation_size=0.1):
    """Prepare the dataset for training, validation, and testing."""
    features, labels = load_dataset(dataset_dir)
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split into training and temporary test set
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, labels, test_size=test_size + validation_size, random_state=42
    )
    
    # Split the temporary test set into validation and test sets
    validation_ratio = validation_size / (test_size + validation_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-validation_ratio, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

# Example usage
if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_dataset("dataset")
    
    # Save the processed data
    np.save("processed_data/X_train.npy", X_train)
    np.save("processed_data/y_train.npy", y_train)
    np.save("processed_data/X_val.npy", X_val)
    np.save("processed_data/y_val.npy", y_val)
    np.save("processed_data/X_test.npy", X_test)
    np.save("processed_data/y_test.npy", y_test)
    
    # Save the scaler for future use
    import pickle
    with open("processed_data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
