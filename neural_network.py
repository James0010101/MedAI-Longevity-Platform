import numpy as np

def relu(x):
    return np.maximum(0.01 * x, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=8, output_size=1, learning_rate=0.03):
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        
        max_clip = 1.0
        dW2 = np.clip(dW2, -max_clip, max_clip)
        db2 = np.clip(db2, -max_clip, max_clip)
        dW1 = np.clip(dW1, -max_clip, max_clip)
        db1 = np.clip(db1, -max_clip, max_clip)
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train_step(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)
        
        epsilon = 1e-15
        loss = -np.mean(y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon))
        return loss
