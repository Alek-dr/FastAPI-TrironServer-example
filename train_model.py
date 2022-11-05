import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def train_network(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs,
    train_losses,
    test_losses,
):
    for epoch in range(num_epochs):
        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # forward feed
        output_train = model(X_train)

        # calculate the loss
        loss_train = criterion(output_train, y_train)

        # backward propagation: calculate gradients
        loss_train.backward()

        # update the weights
        optimizer.step()

        output_test = model(X_test)
        loss_test = criterion(output_test, y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}"
            )


def make_data(n_features: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        random_state=1,
        n_clusters_per_class=1,
        n_classes=n_classes,
        scale=10,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return X, y


def visualize_data(X, y):
    label_colors = ["b", "r"]
    for label, c in enumerate(label_colors):
        ind = y == label
        x_label = X[ind]
        plt.plot(x_label[:, 0], x_label[:, 1], "o", color=c)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()


def train_pytorch_model(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]

    trainset = dataset(X_train, y_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

    model = NeuralNetworkClassificationModel(input_dim)
    learning_rate = 0.01
    epochs = 700
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    loss = 0
    acc = 0
    for i in range(epochs):
        for j, (x_train, y_train) in enumerate(trainloader):
            # calculate output
            output = model(x_train)

            # calculate loss
            loss = loss_fn(output, y_train.reshape(-1, 1))

            # accuracy
            predicted = model(torch.tensor(x_train, dtype=torch.float32))
            acc = (
                predicted.reshape(-1).detach().numpy().round() == y_train.numpy()
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 50 == 0:
            print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))
    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(X_test, dtype=torch.float32))
        test_acc = (predicted.reshape(-1).detach().numpy().round() == y_test).mean()
        print("Test accuracy : {}".format(test_acc))

    example_forward_input = torch.rand(1, 2)
    module = torch.jit.trace(model, example_forward_input)
    torch.jit.save(module, "models/torch/1/model.pt")


"""
Гененрируем случайные данные, обучаем и сохраняем классификаторы
"""

if __name__ == "__main__":
    n_features = 2
    n_classes = 2
    X, y = make_data(n_features, n_classes)
    visualize_data(X, y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_ = scaler.transform(np.array([[10, 4], [40, -1]]).astype(np.float32))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    # Train sklearn model
    clf = RandomForestClassifier(n_estimators=10, max_depth=3)
    clf.fit(X_train, y_train)
    print(f"Random Forest score: {clf.score(X_test, y_test)}")
    with open("models/rf/1/model.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Train xgboost
    bst = XGBClassifier(
        n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic"
    )
    bst.fit(X_train, y_train)
    pred = bst.predict(X_test)
    print(f"Random Forest score: {accuracy_score(pred, y_test)}")
    bst.save_model("models/xgb/1/xgboost.json")

    # Train torch model
    train_pytorch_model(X_train, y_train, X_test, y_test)

    with open("app/prediction_models/data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
