''' This is the code for NPynn v.1.1 '''
import numpy as np



# Dense 
class Dense:

    # Initialize
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs
        
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


        
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient
        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout
class Dropout:

    # Initialize
    def __init__(self, rate):
        
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        
        if not training:
            self.output = inputs.copy()
            return

        
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        
        self.output = inputs * self.binary_mask


    # Backward pass
    def backward(self, dvalues):
        
        self.dinputs = dvalues * self.binary_mask


# Input Layer
class Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs


# ReLU
class ReLU:

    # Forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs
        
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()

        
        self.dinputs[self.inputs <= 0] = 0

    # Calclutate preds
    def predictions(self, outputs):
        return outputs


# Softmax activation
class Softmax:

    # Forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs

        
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # Normalize 
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        
        self.dinputs = np.empty_like(dvalues)

        
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            
            single_output = single_output.reshape(-1, 1)
            
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    # Calculate preds
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Sigmoid activation
class Sigmoid:

    # Forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate preds
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


# Linear activation
class Linear:

    # Forward pass
    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()

    # Calculate preds
    def predictions(self, outputs):
        return outputs


# SGD optimizer
class SGD:

    
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call before param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update params
    def update_params(self, layer):

        
        if self.momentum:

            
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)

                
                layer.bias_momentums = np.zeros_like(layer.biases)

            
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        
        layer.weights += weight_updates
        layer.biases += bias_updates

    
    def post_update_params(self):
        self.iterations += 1


# Adagrad optimizer
class Adagrad:

    # Initialize
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon


    # Call before param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call after param updates
    def post_update_params(self):
        self.iterations += 1


# RMSprop optimizer
class RMSprop:

    # Initialize
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho


    # Call before param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call after param updates
    def post_update_params(self):
        self.iterations += 1


# Adam optimizer
class Adam:

    # Initialize 
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    # Call before param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)

        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call after param updates
    def post_update_params(self):
        self.iterations += 1


# Common loss class
class Loss:

    # Regularization loss 
    def regularization_loss(self):

        
        regularization_loss = 0

        # Calculate regularization loss
        for layer in self.trainable_layers:

            
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))

            
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * \
                                              layer.weights)

            
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                       np.sum(np.abs(layer.biases))

            
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                       np.sum(layer.biases * \
                                              layer.biases)

        return regularization_loss


    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    
    def calculate(self, output, y, *, include_regularization=False):

        # Sample losses calculation
        sample_losses = self.forward(output, y)

        # Mean loss calculation
        data_loss = np.mean(sample_losses)

        
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        
        if not include_regularization:
            return data_loss

        
        return data_loss, self.regularization_loss()

    
    def calculate_accumulated(self, *, include_regularization=False):

        # Mean loss calculation
        data_loss = self.accumulated_sum / self.accumulated_count

        
        if not include_regularization:
            return data_loss

        
        return data_loss, self.regularization_loss()

    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0



# Cross-entropy loss
class CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        
        samples = len(y_pred)

        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true]

        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        
        samples = len(dvalues)
        
        labels = len(dvalues[0])

        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        
        self.dinputs = -y_true / dvalues
        
        self.dinputs = self.dinputs / samples

# Softmax Activation w/ Categorical Crossentropy Loss
class Softmax_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues, y_true):

        
        samples = len(dvalues)

        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        
        self.dinputs = dvalues.copy()
        
        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs = self.dinputs / samples


# Binary cross-entropy loss
class BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        
        samples = len(dvalues)
        
        outputs = len(dvalues[0])

        
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        self.dinputs = self.dinputs / samples


# Mean Squared Error loss
class MSE(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Loss calculation
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        
        samples = len(dvalues)
        
        outputs = len(dvalues[0])

        
        self.dinputs = -2 * (y_true - dvalues) / outputs
        
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss
class MAE(Loss):  # L1 loss

    def forward(self, y_pred, y_true):

        # Loss calculation
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        
        return sample_losses


    # Backward pass
    def backward(self, dvalues, y_true):

        
        samples = len(dvalues)
        
        outputs = len(dvalues[0])

        
        self.dinputs = np.sign(y_true - dvalues) / outputs
        
        self.dinputs = self.dinputs / samples


# Common accuracy class
class Accuracy:

    
    def calculate(self, predictions, y):

        
        comparisons = self.compare(predictions, y)

        
        accuracy = np.mean(comparisons)

        
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        
        return accuracy

    
    def calculate_accumulated(self):

        
        accuracy = self.accumulated_sum / self.accumulated_count

        
        return accuracy

    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Classification Accuracy
class Categorical(Accuracy):

    def __init__(self, *, binary=False):
        
        self.binary = binary

    
    def init(self, y):
        pass

    # Compare preds to ground truth
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


# Regression Accuracy
class Regression(Accuracy):

    def __init__(self):
        
        self.precision = None

    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


# Net class
class Net:

    def __init__(self):
        
        self.layers = []
        
        self.softmax_classifier_output = None

    # Add layers to Model
    def add(self, layer):
        self.layers.append(layer)


    
    def compile(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # FInialize Layers
    def finalize(self):

        
        self.input_layer = Input()

        
        layer_count = len(self.layers)

        
        self.trainable_layers = []

        
        for i in range(layer_count):

            
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])


        
        self.loss.remember_trainable_layers(
            self.trainable_layers
        )

        
        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, CategoricalCrossentropy):
            
            self.softmax_classifier_output = \
                Softmax_CategoricalCrossentropy()

    # Training NN
    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=100, validation_data=None):

        
        self.accuracy.init(y)

        
        train_steps = 1

        # Val data
        if validation_data is not None:
            validation_steps = 1

            
            X_val, y_val = validation_data

        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Training Loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'epoch: {epoch}')

            
            self.loss.new_pass()
            self.accuracy.new_pass()

            
            for step in range(train_steps):

                
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Forward pass
                output = self.forward(batch_X, training=True)

                # Loss calculation
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                # Accuracy calculation
                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                # Backward pass
                self.backward(output, batch_y)


                # Update Params
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f}, ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Validation
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f}, ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            
            if validation_data is not None:

                
                self.loss.new_pass()
                self.accuracy.new_pass()

                
                for step in range(validation_steps):

                    
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val


                    
                    else:
                        batch_X = X_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                        batch_y = y_val[
                            step*batch_size:(step+1)*batch_size
                        ]

                    # Forward pass
                    output = self.forward(batch_X, training=False)

                    # Loss calculation
                    self.loss.calculate(output, batch_y)

                    # Accuracy calculation
                    predictions = self.output_layer_activation.predictions(
                                      output)
                    self.accuracy.calculate(predictions, batch_y)

                
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                # Print summary
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

    # Forward pass
    def forward(self, X, training):

        
        self.input_layer.forward(X, training)

        
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        
        return layer.output


    # Backward pass
    def backward(self, output, y):

        
        if self.softmax_classifier_output is not None:
            
            self.softmax_classifier_output.backward(output, y)

            
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        
        self.loss.backward(output, y)

        
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

