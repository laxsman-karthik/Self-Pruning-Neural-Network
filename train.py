import torch
import torch.nn as nn
import torch.optim as optim
from model import PrunableCNN


def compute_sparsity(model, threshold=1e-2):
    gates = model.get_all_gates()
    total = gates.numel()
    pruned = (gates < threshold).sum().item()
    return (pruned / total) * 100


def train_model(lambda_val, train_loader, test_loader, device, epochs=10):
    model = PrunableCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            classification_loss = criterion(outputs, labels)

            gates = model.get_all_gates()
            sparsity_loss = torch.sum(gates)

            loss = classification_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    accuracy = evaluate_model(model, test_loader, device)
    sparsity = compute_sparsity(model)

    return accuracy, sparsity, model


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total