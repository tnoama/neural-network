import numpy as np
import scipy.io

def compute_mse(w, X, y):
    errors = X @ w - y
    return np.mean(errors**2)

def compute_accuracy(X, Y, W, b):
    X_soft = softmax(np.dot(X, W) + b)  # Compute probabilities
    class_predictions = np.argmax(X_soft, axis=1)  # Get class predictions
    correct = np.sum(class_predictions == Y)
    accuracy = correct / Y.shape[0]
    return accuracy

def get_samples(X, Y, n_samples):
    idxs = np.random.choice(X.shape[0], min(n_samples,X.shape[0]), replace=False)
    return X[idxs], Y[idxs]

def load_data(path):
    dataset = scipy.io.loadmat(path)
    train_data = dataset['Yt'].T  
    val_data = dataset['Yv'].T   
    train_labels = np.argmax(dataset['Ct'], axis=0)  
    val_labels = np.argmax(dataset['Cv'], axis=0)    
    return train_data, train_labels, val_data, val_labels

def softmax(X):
    exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_loss(X, y, W, b):
    m = X.shape[0]  # Number of samples
    X_soft = softmax(np.dot(X, W) + b)  # Compute probabilities
    correct_log_probs = -np.log(X_soft[range(m), y])
    loss = np.sum(correct_log_probs) / m
    return loss


def softmax_gradient(X, Y, W, b):
    m = X.shape[0]
    X_soft = softmax(np.dot(X, W) + b)  
    soft_minus_C = X_soft
    soft_minus_C[np.arange(m), Y] -= 1 #substract 1 from the correct class probabilty for each input
    soft_minus_C /= m    

    dW = np.dot(X.T, soft_minus_C)
    db = np.sum(soft_minus_C, axis=0, keepdims=True)

    return dW, db

def calculate_total_params(layers, is_resNet):
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i + 1] + layers[i + 1]
    if is_resNet:
        total += (layers[1] ** 2) * (len(layers) -3)
    return total
