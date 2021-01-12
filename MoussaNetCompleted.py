# # Copyright (c) 2015 Andrej Karpathy
# # License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# # Source: https://cs231n.github.io/neural-networks-case-study/
# def spiral_data(samples, classes):
#     X = np.zeros((samples * classes, 2))
#     y = np.zeros(samples * classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(samples * class_number, samples * (class_number + 1))
#         r = np.linspace(0.0, 1, samples)
#         t = np.linspace(class_number * 4, (class_number + 1) * 4,
#                         samples) + np.random.randn(samples) * 0.2
#         X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
#         y[ix] = class_number
#     return X, y
#
#
# class Loss:
#     def regularization_loss(self):
#         regularization_loss = 0
#         for layer in self.trainable_layers:
#
#             if layer.weight_regularizer_l1 > 0:
#                 regularization_loss += layer.weight_regularizer_l1 * \
#                     np.sum(np.abs(layer.weights))
#             if layer.weight_regularizer_l2 > 0:
#                 regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights *
#                                                                             layer.weights)
#
#             if layer.bias_regularizer_l1 > 0:
#                 regularization_loss += layer.bias_regularizer_l1 * \
#                     np.sum(np.abs(layer.biases))
#
#             if layer.bias_regularizer_l2 > 0:
#                 regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases *
#                                                                           layer.biases)
#
#         return regularization_loss
#
#     def remember_trainable_layers(self, trainable_layers):
#         self.trainable_layers = trainable_layers
#
#     def calculate(self, output, y, *, include_regularization=False):
#
#         sample_losses = self.forward(output, y)
#         data_loss = np.mean(sample_losses)
#
#         if not include_regularization:
#             return data_loss
#
#         return data_loss, self.regularization_loss()
#
#
# class Loss_MeanSquaredError(Loss):
#     def forward(self, y_pred, y_true):
#         sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
#
#         return sample_losses
#
#     def backward(self, dvalues, y_true):
#         samples = len(dvalues)
#         outputs = len(dvalues[0])
#
#         self.dinputs = -2 * (y_true - dvalues) / outputs
#         self.dinputs = self.dinputs / samples
#
#
# class Loss_CategoricalCrossEntropy(Loss):
#     def forward(self, y_pred, y_true):
#         samples = len(y_pred)
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
#
#         if len(y_true.shape) == 1:
#             correct_confidences = y_pred_clipped[range(samples), y_true]
#         elif len(y_true.shape) == 2:
#             correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
#
#         correct_confidences_clipped = np.clip(
#             correct_confidences, 1e-7, 1 - 1e-7)
#         negative_log_likelihoods = -np.log(correct_confidences_clipped)
#         return negative_log_likelihoods
#
#     def backward(self, dvalues, y_true):
#         samples = len(dvalues)
#         labels = len(dvalues[0])
#
#         if len(y_true.shape) == 1:
#             y_true = np.eye(labels)[y_true]
#
#         self.dinputs = -y_true / dvalues
#         self.dinputs = self.dinputs / samples
#
#
# class Loss_BinaryCrossentropy(Loss):
#     def forward(self, y_pred, y_true):
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
#
#         sample_losses = -(y_true * np.log(y_pred_clipped) +
#                           (1 - y_true) * np.log(1 - y_pred_clipped))
#         sample_losses = np.mean(sample_losses, axis=-1)
#         return sample_losses
#
#     def backward(self, dvalues, y_true):
#         samples = len(dvalues)
#         outputs = len(dvalues[0])
#
#         clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
#         self.dinputs = -(y_true / clipped_dvalues -
#                          (1 - y_true) / (1 - clipped_dvalues)) / outputs
#         self.dinputs = self.dinputs / samples
#
#
# class Activation_Softmax_Loss_CategoricalCrossEntropy():
#
#     def backward(self, dvalues, y_true):
#         samples = len(dvalues)
#         if len(y_true.shape) == 2:
#             y_true = np.argmax(y_true, axis=1)
#         self.dinputs = dvalues.copy()
#         self.dinputs[range(samples), y_true] -= 1
#         self.dinputs = self.dinputs / samples
#
#
# class Optimizer_SGD:
#     def __init__(self, learning_rate=1., decay=0., momentum=0.):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0
#         self.momentum = momentum
#
#     def pre_update_params(self):
#         if self.decay:
#             self.current_learning_rate = self.learning_rate * \
#                 (1. / (1. + self.decay * self.iterations))
#
#     def update_params(self, layer):
#
#         if self.momentum:
#             if not hasattr(layer, 'weight_momentums'):
#                 layer.weight_momentums = np.zeros_like(layer.weights)
#                 layer.bias_momentums = np.zeros_like(layer.biases)
#
#             weight_updates = self.momentum * layer.weight_momentums - \
#                 self.current_learning_rate * layer.dweights
#
#             layer.weight_momentums = weight_updates
#
#             bias_updates = self.momentum * layer.bias_momentums - \
#                 self.current_learning_rate * layer.dbiases
#
#             layer.bias_momentums = bias_updates
#         else:
#
#             weight_updates = -self.current_learning_rate * layer.dweights
#             bias_updates = -self.current_learning_rate * layer.dbiases
#
#         layer.weights += weight_updates
#         layer.biases += bias_updates
#
#     def post_update_params(self):
#         self.iterations += 1
#
#
# class Optimizer_Adam:
#
#     def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0
#         self.epsilon = epsilon
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#
#     def pre_update_params(self):
#         if self.decay:
#             self.current_learning_rate = self.learning_rate * \
#                 (1. / (1. + self.decay * self.iterations))
#
#     def update_params(self, layer):
#         if not hasattr(layer, 'weight_cache'):
#             layer.weight_momentums = np.zeros_like(layer.weights)
#             layer.weight_cache = np.zeros_like(layer.weights)
#             layer.bias_momentums = np.zeros_like(layer.biases)
#             layer.bias_cache = np.zeros_like(layer.biases)
#
#         layer.weight_momentums = self.beta_1 * \
#             layer.weight_momentums + (1 - self.beta_1) * layer.dweights
#         layer.bias_momentums = self.beta_1 * \
#             layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
#
#         weight_momentums_corrected = layer.weight_momentums / \
#             (1 - self.beta_1 ** (self.iterations + 1))
#
#         bias_momentums_corrected = layer.bias_momentums / \
#             (1 - self.beta_1 ** (self.iterations + 1))
#
#         layer.weight_cache = self.beta_2 * layer.weight_cache + \
#             (1 - self.beta_2) * layer.dweights ** 2
#
#         layer.bias_cache = self.beta_2 * layer.bias_cache + \
#             (1 - self.beta_2) * layer.dbiases ** 2
#
#         weight_cache_corrected = layer.weight_cache / \
#             (1 - self.beta_2 ** (self.iterations + 1))
#
#         bias_cache_corrected = layer.bias_cache / \
#             (1 - self.beta_2 ** (self.iterations + 1))
#
#         layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) +
#                                                                                      self.epsilon)
#         layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) +
#                                                                                   self.epsilon)
#
#     def post_update_params(self):
#         self.iterations += 1
#
#
# class Accuracy:
#
#     def calculate(self, predictions, y):
#         comparisons = self.compare(predictions, y)
#         accuracy = np.mean(comparisons)
#
#         return accuracy
#
#
# class Accuracy_Regression(Accuracy):
#     def __init__(self):
#         self.precision = None
#
#     def init(self, y, reinit=False):
#         if self.precision is None or reinit:
#             self.precision = np.std(y) / 250
#
#     def compare(self, predictions, y):
#         return np.absolute(predictions - y) < self.precision
#
#
# class Accuracy_Categorical(Accuracy):
#     def __init__(self, *, binary=False):
#         self.binary = binary
#
#     def init(self, y):
#         pass
#
#     def compare(self, predictions, y):
#         if not self.binary and len(y.shape) == 2:
#             y = np.argmax(y, axis=1)
#         return predictions == y
#
#
# class MoussaNet:
#     def __init__(self):
#         self.layers = []
#         self.softmax_classifier_output = None
#
#     def add(self, layer):
#         self.layers.append(layer)
#
#     def set(self, *, loss, optimizer, accuracy):
#         self.loss = loss
#         self.optimizer = optimizer
#         self.accuracy = accuracy
#
#     def finalize(self):
#
#         self.input_layer = Layer_Input()
#         layer_count = len(self.layers)
#         self.trainable_layers = []
#         for i in range(layer_count):
#
#             if i == 0:
#                 self.layers[i].prev = self.input_layer
#                 self.layers[i].next = self.layers[i + 1]
#
#             elif i < layer_count - 1:
#                 self.layers[i].prev = self.layers[i - 1]
#                 self.layers[i].next = self.layers[i + 1]
#
#             else:
#                 self.layers[i].prev = self.layers[i - 1]
#                 self.layers[i].next = self.loss
#                 self.output_layer_activation = self.layers[i]
#
#             if hasattr(self.layers[i], 'weights'):
#                 self.trainable_layers.append(self.layers[i])
#
#             self.loss.remember_trainable_layers(self.trainable_layers)
#
#     def train(self, X, y, *, epochs=1, print_every=1,
#               validation_data=None):
#
#         self.accuracy.init(y)
#
#         for epoch in range(1, epochs + 1):
#             output = self.forward(X, training=True)
#
#             data_loss, regularization_loss = self.loss.calculate(
#                 output, y, include_regularization=True)
#             loss = data_loss + regularization_loss
#
#             predictions = self.output_layer_activation.predictions(
#                 output)
#             accuracy = self.accuracy.calculate(predictions, y)
#
#             self.backward(output, y)
#             self.optimizer.pre_update_params()
#             for layer in self.trainable_layers:
#                 self.optimizer.update_params(layer)
#             self.optimizer.post_update_params()
#
#             if not epoch % print_every:
#                 print(f'epoch: {epoch}, ' +
#                       f'acc: {accuracy:.3f}, ' +
#                       f'loss: {loss:.3f} (' +
#                       f'data_loss: {data_loss:.3f}, ' +
#                       f'reg_loss: {regularization_loss:.3f}), ' + f'lr: {self.optimizer.current_learning_rate}')
#
#         if validation_data is not None:
#             X_val, y_val = validation_data
#
#             output = self.forward(X_val, training=False)
#             loss = self.loss.calculate(output, y_val)
#
#             predictions = self.output_layer_activation.predictions(
#                 output)
#             accuracy = self.accuracy.calculate(predictions, y_val)
#
#             print(f'validation, ' +
#                   f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')
#
#         if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
#             self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()
#
#     def forward(self, inputs, training):
#         self.input_layer.forward(X, training)
#
#         for layer in self.layers:
#             layer.forward(layer.prev.output, training)
#
#         return layer.output
#
#     def backward(self, output, y):
#         if self.softmax_classifier_output is not None:
#
#             self.softmax_classifier_output.backward(output, y)
#             self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
#             for layer in reversed(self.layers[:-1]):
#                 layer.backward(layer.next.dinputs)
#             return
#
#         self.loss.backward(output, y)
#
#         for layer in reversed(self.layers):
#             layer.backward(layer.next.dinputs)
#
#
# X, y = spiral_data(samples=100, classes=2)
# X_test, y_test = spiral_data(samples=100, classes=2)
#
# y = y.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
#
# model = MoussaNet()
# model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
#                       bias_regularizer_l2=5e-4))
# model.add(Layer_Dense(512, 3))
# model.add(Activation_ReLU())
# model.add(Layer_Dropout(0.1))
# model.add(Activation_Softmax())
#
# model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
#           accuracy=Accuracy_Categorical())
#
# model.finalize()
# model.train(X, y, validation_data=(X_test, y_test),
#             epochs=10000, print_every=100)
#
