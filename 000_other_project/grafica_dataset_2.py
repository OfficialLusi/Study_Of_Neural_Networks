import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# Defining the dense layer class
class Layer_Dense:
    # Initializing the constructor
    def __init__(self, n_inputs, n_neurons):
        # Randomly initializing weights with small values and biases with zeros
        # Weights matrix of shape (n_inputs, n_neurons)
        self.weights = 0.010 * np.random.randn(n_inputs, n_neurons)
        # Biases matrix of shape (1, n_neurons) initialized to zero
        self.biases = np.zeros((1, n_neurons))

    # Defining the forward pass method
    def forward(self, inputs):
        # Store the inputs for use in the backward pass
        self.inputs = inputs
        # Calculate the output of the layer: dot product of inputs and weights, plus biases
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    # Defining the backward pass method
    def backward(self, dvalues):
        # Calculate the gradient of the weights
        # This is done by the dot product of the transposed inputs and the gradient of the output
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Calculate the gradient of the biases
        # This is done by summing the gradient of the output over all samples
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Calculate the gradient of the inputs
        # This is done by the dot product of the gradient of the output and the transposed weights
        self.dinputs = np.dot(dvalues, self.weights.T)


# Defining ReLU activation function class
class Activation_ReLU:
    # Defining the forward pass method for ReLU
    def forward(self, inputs):
        # Calculate the ReLU activation: output is max(0, input)
        self.output = np.maximum(0, inputs)
        # Store the input values for use in the backward pass
        self.inputs = inputs
        return self.output
    
    # Defining the backward pass method for ReLU
    def backward(self, dvalues):
        # Create a copy of the gradient from the next layer
        self.dinputs = dvalues.copy()
        # Zero out gradients where the input values were less than or equal to zero
        self.dinputs[self.inputs <= 0] = 0


# Defining Softmax activation function class
class Activation_Softmax:
    # Defining the Softmax forward pass
    def forward(self, inputs):
        # Store the input values for use in the backward pass
        self.inputs = inputs
        # Compute the exponentiated values after subtracting the max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the exponentiated values to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Store the output probabilities
        self.output = probabilities
    
    # Defining the Softmax backward pass
    def backward(self, dvalues):
        # Initialize the array for the gradients with the same shape as dvalues
        self.dinputs = np.empty_like(dvalues)
        # Iterate over the output and gradient values for each sample
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape the single output to a column vector
            single_output = single_output.reshape(-1, 1)
            # Compute the Jacobian matrix of the softmax output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Compute the sample-wise gradient and store it in dinputs
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Defining a generic Loss function class
class Loss:
    # Method to calculate the loss
    def calculate(self, output, y):
        # Call the forward method to compute the loss for each sample
        sample_losses = self.forward(output, y)
        # Compute the mean of the sample losses to get the overall data loss
        data_loss = np.mean(sample_losses)
        # Return the overall data loss
        return data_loss

    
# Defining the Categorical Cross-Entropy Loss function class
class Loss_CategoricalCrossEntropy(Loss):
    
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples
        samples = len(y_pred)
        
        # Clip the predictions to prevent log(0) which is undefined
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # If the true labels are in a categorical format (not one-hot encoded)
        if len(y_true.shape) == 1:
            # Use advanced indexing to select the predicted probabilities of the correct classes
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        # If the true labels are one-hot encoded
        elif len(y_true.shape) == 2:
            # Multiply the clipped predictions by the true labels and sum along the columns
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Compute the negative log likelihoods
        negative_log_likelihoods = -np.log(correct_confidences)
        
        # Return the computed losses for each sample
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Number of labels in every sample (assuming each sample has the same number of labels)
        labels = len(dvalues[0])
        
        # If the true labels are in a categorical format (not one-hot encoded)
        if len(y_true.shape) == 1:
            # Convert to one-hot encoding
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Combined class for Softmax activation and Categorical Cross-Entropy loss
class Activation_Softmax_Loss_CategoricalCrossEntropy:
    
    # Constructor to initialize the activation and loss objects
    def __init__(self):
        # Create an instance of the Softmax activation function
        self.activation = Activation_Softmax()
        # Create an instance of the Categorical Cross-Entropy loss function
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Perform the forward pass of the Softmax activation function
        self.activation.forward(inputs)
        # Store the output of the Softmax activation
        self.output = self.activation.output
        # Calculate and return the loss using the Categorical Cross-Entropy function
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # If the true labels are one-hot encoded, convert them to class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy the derivative values to avoid modifying the original array
        self.dinputs = dvalues.copy()
        
        # Subtract 1 from the correct class indices
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize the gradient by dividing by the number of samples
        self.dinputs = self.dinputs / samples


# Defining the Stochastic Gradient Descent (SGD) optimizer class
class Optimizer_SGD:
    
    # Constructor to initialize the optimizer with learning rate, decay, and momentum
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        # Set the initial learning rate
        self.learning_rate = learning_rate
        # Initialize the current learning rate with the initial value
        self.current_learning_rate = learning_rate
        # Set the decay rate for the learning rate
        self.decay = decay
        # Initialize the iteration counter
        self.iterations = 0
        # Set the momentum factor
        self.momentum = momentum

    # Method to adjust the learning rate before updating parameters
    def pre_update_params(self):
        # If decay is set, adjust the current learning rate based on the decay rate and the number of iterations
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Method to update the parameters of a given layer
    def update_parameters(self, layer):
        
        # If momentum is used
        if self.momentum:
            # Check if the layer has momentum arrays; if not, initialize them with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Calculate the weight updates using momentum
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            # Update the weight momentums with the new weight updates
            layer.weight_momentums = weight_updates
            
            # Calculate the bias updates using momentum
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            # Update the bias momentums with the new bias updates
            layer.bias_momentums = bias_updates 

        # If momentum is not used, perform standard SGD updates
        else:
            # Calculate the weight updates without momentum
            weight_updates = -self.current_learning_rate * \
                                layer.dweights
            # Calculate the bias updates without momentum
            bias_updates = -self.current_learning_rate * \
                                layer.dbiases
            
        # Update the layer's weights with the calculated weight updates
        layer.weights += weight_updates
        # Update the layer's biases with the calculated bias updates
        layer.biases += bias_updates

    # Method to increment the iteration counter after updating parameters
    def post_update_params(self):
        # Increment the iteration counter by one
        self.iterations += 1


# Adagrad optimizer
class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        # Set the initial learning rate
        self.learning_rate = learning_rate
        # Initialize the current learning rate with the initial value
        self.current_learning_rate = learning_rate
        # Set the decay rate for the learning rate
        self.decay = decay
        # Initialize the iteration counter
        self.iterations = 0
        # Set a small constant to prevent division by zero
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        # If decay is set, adjust the current learning rate based on the decay rate and the number of iterations
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_parameters(self, layer):
        
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        # Increment the iteration counter by one
        self.iterations += 1



# RMSprop optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        # Set the initial learning rate
        self.learning_rate = learning_rate
        # Initialize the current learning rate with the initial value
        self.current_learning_rate = learning_rate
        # Set the decay rate for the learning rate
        self.decay = decay
        # Initialize the iteration counter
        self.iterations = 0
        # Set a small constant to prevent division by zero
        self.epsilon = epsilon
        # Set the decay rate for the moving average
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        # If decay is set, adjust the current learning rate based on the decay rate and the number of iterations
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_parameters(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        # Increment the iteration counter by one
        self.iterations += 1




# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        # Set the initial learning rate
        self.learning_rate = learning_rate
        # Initialize the current learning rate with the initial value
        self.current_learning_rate = learning_rate
        # Set the decay rate for the learning rate
        self.decay = decay
        # Initialize the iteration counter
        self.iterations = 0
        # Set a small constant to prevent division by zero
        self.epsilon = epsilon
        # Set the decay rate for the moving average of the first moment (mean)
        self.beta_1 = beta_1
        # Set the decay rate for the moving average of the second moment (variance)
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        # If decay is set, adjust the current learning rate based on the decay rate and the number of iterations
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_parameters(self, layer):

        # If the layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            # Initialize momentum and cache arrays for weights and biases
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum for weights with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        # Update momentum for biases with current gradients
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum for weights
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Get corrected momentum for biases
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache for weights with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        # Update cache for biases with squared current gradients
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache for weights
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        # Get corrected cache for biases
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        # Increment the iteration counter by one
        self.iterations += 1



# Crea il dataset
X, y = spiral_data(samples=100, classes=3)

# Crea i livelli densi
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7) # 1, 1e-35

losses = []
accuracies = []

for epoch in range(2001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}")

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_params()

# Visualizza i risultati
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Define a function to visualize the decision boundary along with the dataset points
def plot_decision_boundary(X, y, model):
    # Determine the minimum and maximum values for the x-axis with some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # Determine the minimum and maximum values for the y-axis with some padding
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # Create a meshgrid of values from the determined x and y ranges with a step of 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    # Use the model to predict the class for each point in the meshgrid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    # Get the class with the highest probability for each point and reshape to the meshgrid shape
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    # Plot the decision boundary by coloring the regions
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    # Scatter plot the original dataset points on top of the decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k', cmap=plt.cm.Spectral)
    # Set the limits for the x-axis
    plt.xlim(xx.min(), xx.max())
    # Set the limits for the y-axis
    plt.ylim(yy.min(), yy.max())


# Function to perform a forward pass through the layers
def model_forward(X):
    # Perform the forward pass through the first dense layer
    dense1_output = dense1.forward(X)
    # Perform the forward pass through the ReLU activation function
    activation1_output = activation1.forward(dense1_output)
    # Perform the forward pass through the second dense layer
    dense2_output = dense2.forward(activation1_output)
    # Return the output from the second dense layer
    return dense2_output


plt.subplot(2, 2, 3)
plot_decision_boundary(X, y, model_forward)
plt.title('Decision Boundary')

# Function to visualize the decision boundary without dataset points
def plot_decision_boundary_no_points(X, model):
    # Calculate the minimum and maximum values for the x-axis with a margin of 1 unit
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # Calculate the minimum and maximum values for the y-axis with a margin of 1 unit
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Create a meshgrid of points within the x and y axis ranges with a step of 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    # Flatten the grid to pass through the model and obtain the predictions
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    # Reshape the predictions to match the grid shape
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    # Create a filled contour plot of the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    # Set the x-axis limits to the minimum and maximum values
    plt.xlim(xx.min(), xx.max())
    # Set the y-axis limits to the minimum and maximum values
    plt.ylim(yy.min(), yy.max())


plt.subplot(2, 2, 4)
plot_decision_boundary_no_points(X, model_forward)
plt.title('Decision Boundary (No Points)')

plt.tight_layout()
plt.show()
