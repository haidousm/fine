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
