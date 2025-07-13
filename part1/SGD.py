import numpy as np
import matplotlib.pyplot as plt
import Utils as Utils

def run_synthetic_example(m, n, lr=0.1, mini_batch_size=10, epochs= 200):
    print(f"experimenting {m} samples with {n} features")
    
    X, y, sol, lambda_ = setup_synthetic_data(m, n)
    
    loss = synthetic_sgd(X, y, lambda_, lr, mini_batch_size, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss)), loss, label='Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Synthetic SGD', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def setup_synthetic_data(m, n):
    X = np.random.randn(m, n)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.exp(0.3 * np.random.randn(min(m, n)))
    X = U @ np.diag(S) @ Vt
    sol = np.random.randn(n)
    y = X @ sol + 0.05 * np.random.randn(m)  # Add noise to the output
    lambda_ = 0.001
    I_n = np.eye(n)
    sol = np.linalg.solve((1.0 / m) * (X.T @ X) + lambda_ * I_n, (1.0 / m) * X.T @ y)
    return X, y, sol, lambda_

def synthetic_sgd(X, y, lambda_, lr, mini_batch_size, epochs):
    m, n = X.shape
    w = np.zeros(n)
    mini_batch_size = 10
    loss = []

    for epoch in range(1, epochs):
        # Reduce the learning rate every 50 epochs
        if epoch % 50 == 0:
            lr *= 0.5
            print("Learning rate:", lr)

        idxs = np.random.permutation(m)

        for k in range(m // mini_batch_size):
            Ib = idxs[k * mini_batch_size:(k + 1) * mini_batch_size]  
            Xb = X[Ib, :]  
            grad = (1.0 / mini_batch_size) * Xb.T @ (Xb @ w - y[Ib]) + lambda_ * w
            w -= lr * grad
        
        # Compute the MSE for the entire dataset
        mse = Utils.compute_mse(w, X, y)
        loss.append(mse)
    return loss

def best_SGD_params(data_path, lr, batch_size, epochs):
    train_data, train_labels, val_data, val_labels = Utils.load_data(data_path)
    best_avg_val_acc = 0
    best_val_acc_plot = []
    best_train_acc_plot = []
    for i in range(len(lr)):
        for j in range(len(batch_size)):
            train_acc, val_acc, avg_val_acc = sgd(train_data, train_labels, val_data, val_labels, lr[i], batch_size[j], epochs)
            if avg_val_acc > best_avg_val_acc:
                best_avg_val_acc = avg_val_acc
                best_train_acc_plot = train_acc
                best_val_acc_plot = val_acc
                best_lr = lr[i]
                best_batch_size = batch_size[j]
            print()
    print(f"Best validation accuracy: {best_avg_val_acc:.4f} with learning rate: {best_lr} and batch size: {best_batch_size}")            
    plot_accuracies(best_train_acc_plot, best_val_acc_plot, data_path)
    
def sgd(X_train, y_train, X_val, y_val, lr, batch_size, epochs):
    print(f"Training with learning rate: {lr}, batch size: {batch_size}, epochs: {epochs}")
    num_of_tries = 30
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    W = np.random.randn(num_features, num_classes) / num_features
    b = np.zeros((1, num_classes))
    
    train_accuracies = []
    val_accuracies = []
    avg_val_acc = 0
    best_val_acc = 0
    epochs_without_improvement = 0  # Track how many epochs without improvement

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(len(X_train))
        train_data = X_train[shuffled_indices]
        Y = y_train[shuffled_indices]

        for i in range(len(train_data) // batch_size):
            batch_X, batch_Y = get_batch(train_data, Y, batch_size, i)
            dW, db = Utils.softmax_gradient(batch_X, batch_Y, W, b)
            W -= lr * dW
            b -= lr * db

        X_sample, Y_sample = Utils.get_samples(X_train, y_train, batch_size)
        train_acc = Utils.compute_accuracy(X_sample, Y_sample, W, b)
        train_accuracies.append(train_acc)
        
        X_sample, Y_sample = Utils.get_samples(X_val, y_val, batch_size)
        val_acc = Utils.compute_accuracy(X_sample, Y_sample, W, b)
        val_accuracies.append(val_acc)

        # Check for improvement in validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0  
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= num_of_tries:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    avg_val_acc = np.mean(val_accuracies[-10:])
    print(f"Training Accuracy: {train_acc:.4f}, Average Validation Accuracy (last 10 epochs): {avg_val_acc:.4f}")
    return train_accuracies, val_accuracies, avg_val_acc


def plot_accuracies(train_accuracies, val_accuracies, fileName):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy", linewidth=2)
    plt.plot(range(len(val_accuracies)), val_accuracies, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"{fileName} - SGD", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(0, 1)
    
    plt.show()

def get_batch(train_data, y, batch_size, batch_index):
    start = batch_index * batch_size
    end = start + batch_size
    return train_data[start:end], y[start:end]
