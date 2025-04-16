import matplotlib.pyplot as plt

# Example loss values from your training log (for epochs 1 through 10)
epochs = list(range(1, 11))
train_losses = [312.2422, 295.3851, 282.8034, 294.9014, 288.5667, 290.2280, 290.4190, 285.2149, 291.1659, 282.6426]
val_losses   = [74.8631, 62.8364, 66.8538, 68.3658, 58.7833, 62.3657, 65.8065, 60.6454, 67.3443, 67.5027]

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()