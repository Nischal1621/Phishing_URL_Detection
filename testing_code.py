import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc
)
from torch.utils.data import DataLoader, TensorDataset

# Load Testing Dataset (Unseen Inputs)
df_test = pd.read_csv('testing_dataset.csv')
X_test = df_test.drop(['url', 'label'], axis=1).values
y_test = df_test['label'].values

# Standardize Features using the same scaler
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Convert to PyTorch tensors
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define ResMLP Model (Same as Training)
class ResMLP(nn.Module):
    def __init__(self, input_size=56, hidden_size=128, num_classes=2):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        bn1_out = self.bn1(x1)
        out = self.relu1(bn1_out)
        x2 = self.fc2(out)
        bn2_out = self.bn2(x2)
        out = self.relu2(bn2_out + out)
        x3 = self.fc3(out)
        bn3_out = self.bn3(x3)
        out = self.relu3(bn3_out + out)
        out = self.fc_out(out)
        return out

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResMLP().to(device)
model.load_state_dict(torch.load('res_mlp_model.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Initialize Metrics Storage
actual_labels = []
predicted_labels = []
prediction_times = []
probabilities = []

# Perform Testing
print("\nActual vs. Predicted Labels:")
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        start_time = time.time()  # Start timing prediction
        outputs = model(data)
        end_time = time.time()  # End timing prediction

        prediction_times.append(end_time - start_time)
        prob = torch.softmax(outputs, dim=1)[:, 1].item()  # Get probability of class 1
        probabilities.append(prob)

        _, predicted = torch.max(outputs, 1)
        actual_labels.append(target.item())
        predicted_labels.append(predicted.item())

        print(f"Actual: {target.item()}, Predicted: {predicted.item()}")

# Compute Metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)
mcc = matthews_corrcoef(actual_labels, predicted_labels)
conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# Compute Testing Time
avg_test_time = np.mean(prediction_times)

# Print Results
print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Average Testing Time: {avg_test_time:.6f} seconds")

# Save Predictions to Excel
df_predictions = pd.DataFrame({
    'Actual Label': actual_labels,
    'Predicted Label': predicted_labels
})
df_predictions.to_excel('studentname-regnumber-prediction.xlsx', index=False)

# Save Performance Metrics to Excel
df_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'Avg Test Time'],
    'Value': [accuracy, precision, recall, f1, mcc, avg_test_time]
})
df_metrics.to_excel('studentname-regnumber-metrics.xlsx', index=False)

# Save Testing Time per Input
df_test_time = pd.DataFrame({
    'Input Index': range(len(prediction_times)),
    'Testing Time (s)': prediction_times
})
df_test_time.to_excel('studentname-regnumber-testingtime.xlsx', index=False)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(actual_labels, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('studentname-regnumber-rocgraph.jpeg')
plt.show()
