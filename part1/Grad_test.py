import numpy as np
import matplotlib.pyplot as plt


def plot_grad_test(y0, y1,max_iter, title):
    plt.figure()
    plt.semilogy(range(max_iter), y0, label="Zero order approx (O(ϵ))")
    plt.semilogy(range(max_iter), y1, label="First order approx (O(ϵ²))")
    plt.legend()
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.grid()
    plt.show()
    
def softmax_gradient_test(F, g_F, W, b, epsilon=0.05, max_iter=8):
    F0 = F(W, b)                
    g_F_W, g_F_b = g_F(W,b)  

    d_W = np.random.randn(*W.shape)
    d_b = np.random.randn(*b.shape)

    g0_W = np.sum(g_F_W * d_W)
    g0_b = np.sum(g_F_b * d_b)     

    y0 = []  # Errors for zero-order 
    y1 = []  # Errors for first-order 

    print(f"{'k':<3}\t{'error order 0':<20}{'error order 1':<20}") 
    for k in range(max_iter):
        epsk = epsilon * (0.5 ** k)  
        Fk = F(W + epsk * d_W, b + epsk * d_b)
        F1 = F0 + (epsk * (g0_W + g0_b))
        y0.append(abs(Fk - F0))      
        y1.append(abs(Fk - F1))     
        print(f"{k:<3}\t{y0[-1]:<20.6e}{y1[-1]:<20.6e}")  
    plot_grad_test(y0, y1, max_iter, "Softmax Gradient Test")

def gradient_test_layer(F, g_F, x, title, epsilon=0.5, max_iter=8):
    F0 = F(x)
    g_F_0 = g_F(x)
    d = np.random.randn(*x.shape)
    y0 = []  # Errors for zero-order 
    y1 = []  # Errors for first-order

    for k in range(max_iter):
        epsk = epsilon * (0.5 ** k)
        Fk = F(x + epsk * d)
        F1 = F0 + epsk * np.dot(g_F_0.flatten(), d.flatten())
        y0.append(abs(Fk - F0))
        y1.append(abs(Fk - F1))
    plot_grad_test(y0, y1, max_iter, title)

def gradient_test_NN(model, data, label, title, epsilon=0.5, max_iter=20):
    # x is the long vector of all parameters (W and b)
    x = model.get_parameters_vector()
    model.backward(model.forward(data), label)
    # g_F_0 is the long vector of all derivatives of x (dW and db)
    g_F_0 = model.get_derivatives_vector()

    def F(X):
        model.set_parameters_from_vector(X)
        return model.calculate_loss(data, label)

    F0 = F(x)

    d = np.random.randn(*g_F_0.shape)
    y0 = []  # Errors for zero-order 
    y1 = []  # Errors for first-order

    for k in range(max_iter):
        epsk = epsilon * (0.5 ** k)
        Fk = F(x + epsk * d)
        F1 = F0 + epsk * np.dot(g_F_0.flatten(), d.flatten())
        y0.append(abs(Fk - F0))
        y1.append(abs(Fk - F1))
    plot_grad_test(y0, y1, max_iter, title)