import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate random matrices and their eigenvalues
num_samples = 10000
matrix_size = 3
matrices = np.random.randn(num_samples, matrix_size, matrix_size)
eigenvalues = np.linalg.eigvals(matrices)

# Convert data to PyTorch tensors
X = torch.tensor(matrices.reshape(num_samples, -1), dtype=torch.float32)
y = torch.tensor(eigenvalues.real, dtype=torch.float32)

# Split data into training and test sets
train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the model
class EigenvaluePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EigenvaluePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
input_size = matrix_size ** 2
hidden_size = 64
output_size = matrix_size
model = EigenvaluePredictor(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
train_losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
    
# Get the output for a sample matrix input
sample_matrix = np.array([[3, 2, 4], [2, 0, 2], [4, 2, 3]])
sample_input = torch.tensor(sample_matrix.reshape(1, -1), dtype=torch.float32)

with torch.no_grad():
     sample_output = model(sample_input)
     print('Sample Matrix Input:')
     print(sample_matrix)
     print('Predicted Eigenvalues:')
     print(sample_output.squeeze().numpy())

# Plot the training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
