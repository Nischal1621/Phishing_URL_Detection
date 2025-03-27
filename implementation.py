import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load the pre-separated training dataset
df_train = pd.read_csv('training_dataset.csv')
X_train = df_train.drop(['url', 'label'], axis=1).values
y_train = df_train['label'].values

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define ResMLP Model with Batch Normalization
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
        return out, [bn1_out, bn2_out, bn3_out]

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Display Number of Parameters per Layer
print("\nNumber of Parameters per Layer:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()}")

# Training setup
num_epochs = 5
batch_interval = 100
loss_values = []
accuracy_values = []
batch_norm_stats = {1: [], 2: [], 3: []}  # Store mean BN outputs
bn_featurewise = {batch_idx: [] for batch_idx in range(10)}

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs, bn_outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy for this batch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

        # Store batch normalization outputs
        if batch_idx < 10:
            first_sample_bn = [bn[0].detach().cpu().numpy() for bn in bn_outputs]
            bn_featurewise[batch_idx] = first_sample_bn

        if batch_idx % batch_interval == 0:
            for i, bn_out in enumerate(bn_outputs):
                batch_norm_stats[i+1].append(bn_out.mean().item())

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    loss_values.append(avg_loss)
    accuracy_values.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Save Loss and Accuracy Data
df_loss = pd.DataFrame({'Epoch': range(1, num_epochs+1), 'Loss': loss_values})
df_loss.to_excel('loss.xlsx', index=False)
df_acc = pd.DataFrame({'Epoch': range(1, num_epochs+1), 'Accuracy': accuracy_values})
df_acc.to_excel('accuracy.xlsx', index=False)

# Plot Loss Graph
plt.figure()
plt.plot(range(1, num_epochs+1), loss_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_graph.jpeg')
plt.show()

# Plot Accuracy Graph
plt.figure()
plt.plot(range(1, num_epochs+1), accuracy_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.savefig('accuracy_graph.jpeg')
plt.show()

# Plot Batch Normalization Feature-wise Graphs for First Sample in First 10 Batches
for batch_idx in range(10):
    bn1, bn2, bn3 = bn_featurewise[batch_idx]
    features = range(len(bn1))
    plt.figure()
    plt.plot(features, bn1, label='BN Layer 1', marker='o')
    plt.plot(features, bn2, label='BN Layer 2', marker='s')
    plt.plot(features, bn3, label='BN Layer 3', marker='^')
    plt.xlabel('Features')
    plt.ylabel('Normalized Value')
    plt.title(f'Batch {batch_idx+1}: First Sample BN Output')
    plt.legend()
    plt.savefig(f'bn_output_batch_{batch_idx+1}.jpeg')
    plt.show()

# Save Model
torch.save(model.state_dict(), 'res_mlp_model.pth')
print("Model saved to 'res_mlp_model.pth'.")