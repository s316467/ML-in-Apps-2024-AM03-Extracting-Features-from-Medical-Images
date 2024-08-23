import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset.ROIPatchedDataset import ROIPatchDataset


class ROIRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ROIRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@staticmethod
def train_and_evaluate_regressor(
    train_features,
    train_coords,
    test_features,
    test_coords,
    input_dim,
    output_dim,
    epochs=100,
):
    model = ROIRegressor(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_coords)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features)
        test_loss = criterion(test_outputs, test_coords)
        print(f"Test Loss: {test_loss.item():.4f}")

    return model, test_loss.item()
