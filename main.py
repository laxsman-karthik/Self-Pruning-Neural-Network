import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train import train_model


def plot_gate_distribution(model):
    gates = model.get_all_gates().cpu().detach().numpy()

    plt.figure(figsize=(10, 6))

    plt.hist(gates, bins=100, density=True)

    plt.yscale('log')

    plt.title("Gate Value Distribution (Log Scale)", fontsize=14)
    plt.xlabel("Gate Value", fontsize=12)
    plt.ylabel("Density (log scale)", fontsize=12)

    plt.axvline(x=0.01, linestyle='--', label='Pruning Threshold (0.01)')
    plt.axvline(x=0.5, linestyle='--', label='Mid Gate Value')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    lambda_values = [0.0001, 0.001, 0.01]

    results = []
    best_model = None
    best_acc = 0

    for lam in lambda_values:
        print(f"\nTraining with lambda = {lam}")

        acc, sparsity, model = train_model(
            lam, train_loader, test_loader, device
        )

        results.append((lam, acc, sparsity))

        print(f"Lambda: {lam}, Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("\nFinal Results:")
    print("Lambda | Accuracy | Sparsity")
    for r in results:
        print(f"{r[0]} | {r[1]:.2f}% | {r[2]:.2f}%")

    plot_gate_distribution(best_model)


if __name__ == "__main__":
    main()