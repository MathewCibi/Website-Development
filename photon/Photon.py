import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
# import torch_directml

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
#input_size = 784 # 28 * 28
input_size = 28
sequence_length = 28
hidden_size = 128
num_classes = 62
num_layers = 2
num_epochs = 50
batch_size = 100
lr = 1e-3

# Mnist
train_dataset = torchvision.datasets.EMNIST(root="./data",
                                           train=True, split="balanced",
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.EMNIST(root="./data",
                                           train=False, split="balanced",
                                           transform=transforms.ToTensor(),
                                           download=True)
if __name__ == '__main__':

  # Data Loader
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

  class RNN(nn.Module):
      def __init__(self, input_size, hidden_size, num_classes, num_layers) -> None:
          super().__init__()
          self.num_layers = num_layers
          self.hidden_size = hidden_size
          self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
          self.fc1 = nn.Linear(hidden_size, 256)
          self.bn1 = nn.BatchNorm1d(256)
          self.relu = nn.ReLU(inplace=True)
          self.dropout = nn.Dropout(p=0.5)
          self.fc2 = nn.Linear(256, num_classes)
          
      def forward(self, x):
          h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
          out, _ = self.gru(x, h0)
          out = out[:, -1, :]
          out = self.fc1(out)
          out = self.bn1(out)
          out = self.relu(out)
          out = self.dropout(out)
          out = self.fc2(out)
          return out

  model = RNN(input_size, hidden_size, num_classes, num_layers).to(device)
  model.load_state_dict(torch.load('photon.pth'))

  #img = Image.open('b_small.png')
  #t = transforms.Grayscale()
  #img_tensor = t(img)
  #img_tensor = transforms.ToTensor()(img_tensor).to(device)
  #print(img_tensor)
  #model.eval()
  #y_pred = model(img_tensor)
  #_, out = torch.max(y_pred.data, 1)
  #print(out)
  #print(train_dataset.classes[out.item()])
  img, label = next(iter(train_dataset))
  img = img.permute(2, 1, 0)
  plt.imshow(img)
  plt.show()
  
  def train():

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Train the model
    n_total_steps = len(train_loader)
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            # print(images.shape)
            X, y = images.reshape(-1, sequence_length, input_size).to(device), labels.to(device)
            y_pred = model(X)
            loss = criterion(y_pred[:100], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch: {epoch+1} | Loss: {loss.item():.4f} | Step: {((i+1)/n_total_steps):.4f}')
  train()

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
      print(f'Accuracy of the network on {len(test_loader)} test images: {acc} %')

  torch.save(model.state_dict(), "photon.pth")