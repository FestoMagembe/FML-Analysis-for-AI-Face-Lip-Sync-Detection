import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# evaluate.py (continued)
import seaborn as sns

def evaluate_model(model_path, X_test, y_test):
   """Evaluate the trained model on the test set."""
   # Load the model
   model = load_model(model_path)
   
   # Make predictions
   y_pred_prob = model.predict(X_test)
   y_pred = (y_pred_prob > 0.5).astype(int)
   
   # Calculate metrics
   print("Classification Report:")
   print(classification_report(y_test, y_pred))
   
   # Confusion Matrix
   cm = confusion_matrix(y_test, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.title('Confusion Matrix')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.savefig("evaluation/confusion_matrix.png")
   plt.close()
   
   # ROC Curve
   fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
   roc_auc = auc(fpr, tpr)
   
   plt.figure(figsize=(8, 6))
   plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic')
   plt.legend(loc="lower right")
   plt.savefig("evaluation/roc_curve.png")
   plt.close()
   
   return {
       'accuracy': model.evaluate(X_test, y_test)[1],
       'roc_auc': roc_auc
   }

# Example usage
if __name__ == "__main__":
   # Create output directory
   import os
   os.makedirs("evaluation", exist_ok=True)
   
   # Load test data
   X_test = np.load("processed_data/X_test.npy")
   y_test = np.load("processed_data/y_test.npy")
   
   # Evaluate the model
   metrics = evaluate_model("models/lip_sync_model_best.h5", X_test, y_test)
   print(f"Test Accuracy: {metrics['accuracy']:.4f}")
   print(f"ROC AUC: {metrics['roc_auc']:.4f}")
