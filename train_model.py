import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import SimpleCNN
from utils.dataset_loader import load_mnist_data

def train_model(epochs=5):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = load_mnist_data(batch_size=64, train=True)

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'models/pretrained_model.pth')
    print('Model saved to models/pretrained_model.pth')

if __name__ == '__main__':
    train_model()
