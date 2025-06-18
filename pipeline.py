import pandas as pd
import yaml
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from model_utils import create_pipeline

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize Weights & Biases
wandb.init(project=config['wandb']['project'], name=config['wandb']['experiment_name'], config=config)

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = create_pipeline(config)

def log_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Log image to wandb
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


# Train model
pipeline.fit(X_train, y_train)

# Evaluate with cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)
wandb.log({"cv_accuracy": cv_scores.mean()})

# Test set performance
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
wandb.log({"test_accuracy": report['accuracy']})
wandb.log({
    "macro_precision": report['macro avg']['precision'],
    "macro_recall": report['macro avg']['recall']
})

# Log confusion matrix
log_confusion_matrix(y_test, y_pred, iris.target_names)
wandb.log({"classification_report": report})

# Save model
joblib.dump(pipeline, "iris_pipeline_model.pkl")
wandb.save("iris_pipeline_model.pkl")

print("✅ Training complete and model saved with W&B tracking.")
