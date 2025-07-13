import numpy as np
import part1.Grad_test as grad_test

def jac_test_layer(in_dim, out_dim, by_param):
    W_layer, W2_layer, b_layer = initialize_weight_and_bias(in_dim, out_dim)
    X_rand = np.random.randn(1, in_dim)
    u = np.random.randn(out_dim)
    match by_param:
        case 'W':            
            def g(W):
                X_next = np.dot(X_rand, W) + b_layer
                X_next = np.tanh(X_next)
                g_X_u = np.dot(X_next, u)
                return g_X_u
            
            def gradient_g(W):
                X_next = np.dot(X_rand, W) + b_layer
                sigma_prime = 1 - np.tanh(X_next) ** 2
                sigma_prime_u = sigma_prime * u
                grad_W = np.dot(X_rand.T, sigma_prime_u) / X_rand.shape[0]
                return grad_W
            
            grad_test.gradient_test_layer(g, gradient_g, W_layer, 'Jacobian Test for W')
        case 'b':
            def g(b):
                X_next = np.dot(X_rand, W_layer) + b
                X_next = np.tanh(X_next)
                g_X_u = np.dot(X_next, u)
                return g_X_u
            
            def gradient_g(b):
                X_next = np.dot(X_rand, W_layer) + b
                sigma_prime = 1 - np.tanh(X_next) ** 2
                sigma_prime_u = sigma_prime * u
                grad_b = np.sum(sigma_prime_u, axis=0, keepdims=True) / X_rand.shape[0]
                return grad_b
            
            grad_test.gradient_test_layer(g, gradient_g, b_layer, 'Jacobian Test for b')

def jac_test_resnet_layer(dim, by_param):
    W_layer, W2_layer, b_layer = initialize_weight_and_bias(dim, dim)
    X_rand = np.random.randn(1, dim)
    u = np.random.randn(dim)

    match by_param:
        case 'W1':
            def g(W):
                X_next = np.dot(X_rand, W) + b_layer
                X_next = np.tanh(X_next)
                X_next = X_rand + np.dot(X_next, W2_layer)
                g_X_u = np.dot(X_next, u)
                return g_X_u
            
            def gradient_g(W):
                X_next = np.dot(X_rand, W) + b_layer
                sigma_prime = 1 - np.tanh(X_next) ** 2
                sigma_prime_W2T_u = sigma_prime * np.dot(u, W2_layer.T)
                grad_W = np.dot(X_rand.T, sigma_prime_W2T_u) / X_rand.shape[0]
                return grad_W
            
            grad_test.gradient_test_layer(g, gradient_g, W_layer, 'Jacobian Test for W1 - ResNet')

        case 'W2':
                def g(W2):
                    X_next = np.dot(X_rand, W_layer) + b_layer
                    X_next = np.tanh(X_next)
                    X_next = X_rand + np.dot(X_next, W2)
                    g_X_u = np.dot(X_next, u)
                    return g_X_u
                
                def gradient_g(W2):
                    X_next = np.dot(X_rand, W_layer) + b_layer
                    X_next = np.tanh(X_next)
                    X_next = X_rand + np.dot(X_next, W2)
                    grad_W2 = np.dot(X_next.T, u.reshape(1, dim)) / X_rand.shape[0]
                    return grad_W2

                grad_test.gradient_test_layer(g, gradient_g, W2_layer, 'Jacobian Test for W2 - ResNet')
        
        case 'b':
            def g(b):
                X_next = np.dot(X_rand, W_layer) + b
                X_next = np.tanh(X_next)
                X_next = X_rand + np.dot(X_next, W2_layer)
                g_X_u = np.dot(X_next, u)
                return g_X_u
            
            def gradient_g(b):
                X_next = np.dot(X_rand, W_layer) + b
                sigma_prime = 1 - np.tanh(X_next) ** 2
                sigma_prime_W2T_u = sigma_prime * np.dot(u, W2_layer.T)
                grad_b = np.sum(sigma_prime_W2T_u, axis=0, keepdims=True) / X_rand.shape[0]
                return grad_b
            
            grad_test.gradient_test_layer(g, gradient_g, b_layer, 'Jacobian Test for b - ResNet')

def initialize_weight_and_bias(in_dim, out_dim):
    W = np.random.randn(in_dim, out_dim) / in_dim
    W2 = np.random.randn(in_dim, out_dim) / in_dim
    b = np.zeros((1, out_dim))
    return W, W2, b