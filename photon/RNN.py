import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch_directml

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
#input_size = 784 # 28 * 28
input_size = 28
sequence_length = 28
hidden_size = 128
num_classes = 10
num_layers = 2
num_epochs = 0
batch_size = 100
lr = 1e-3

# Mnist
train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

# Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Fully connected neural network with one hidden layer.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, sequence_length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        # out: batch_size, sequence_length, hidden_size
        # out (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        out = self.fc(out)
        return out
    
model = RNN(input_size, hidden_size, num_classes, num_layers)
model.load_state_dict(torch.load('model_state.pth'))
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        X, y = images.reshape(-1, sequence_length, input_size).to(device), labels.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch: {epoch+1} | Loss: {loss.item():.4f} | Step: {((i+1)/n_total_steps):.4f}')

with torch.inference_mode():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on {len(train_loader)} test images: {acc} %')
torch.save(model.state_dict(), "model_state.pth")