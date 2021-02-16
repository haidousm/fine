import copy
import gzip
import pickle
import time

import numpy as np

from .model_utils import Input


class Sequential:

    def __init__(self, layers=[], loss=None, optimizer=None, accuracy=None):

        self.layers = layers
        self.set(loss=loss, optimizer=optimizer, accuracy=accuracy)
        self.softmax_classifier_output = None
        self.first_run = True

    def add(self, layer):

        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):

        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Input()

        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:

                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:

                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:

                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, X, y, *, epochs=1,
              batch_size=None, print_every=None,
              validation_data=None):

        if self.first_run:
            self.finalize()
            self.first_run = False

        training_start_time = time.time()
        self.accuracy.init(y)
        train_steps = 1

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

        for epoch in range(1, epochs + 1):

            self.loss.new_pass()
            self.accuracy.new_pass()
            epoch_start_time = time.time()
            step_start_time = time.time()
            for step in range(train_steps):

                if batch_size is None:

                    batch_X = X
                    batch_y = y

                else:

                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)

                data_loss = self.loss.calculate(output, batch_y)

                loss = data_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if print_every is not None:
                    if not step % print_every and step != 0:
                        step_end_time = round(time.time() - step_start_time, 2)
                        step_start_time = time.time()
                        print(
                            f'epoch: {epoch}, ' +
                            f'step: {step}, ' +
                            f'acc: {accuracy:.3f}, ' +
                            f'loss: {loss:.3f}, ' +
                            f'lr: {self.optimizer.current_learning_rate}, ' +
                            f'time: {step_end_time}s'
                        )

            epoch_data_loss = self.loss.calculate_accumulated()

            epoch_loss = epoch_data_loss

            epoch_accuracy = self.accuracy.calculate_accumulated()

            epoch_end_time = round(time.time() - epoch_start_time, 2)

            print(
                f'training - epoch {epoch}, ' +
                f'acc: {epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f} ' +
                f'lr: {self.optimizer.current_learning_rate}, ' +
                f'time: {epoch_end_time}s'
            )

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

        training_end_time = round(time.time() - training_start_time)
        print('-----------------------------------')
        print(f'total time {training_end_time}s')

    def evaluate(self, X_val, y_val, *, epoch=None, batch_size=None):

        if self.first_run:
            self.finalize()
            self.first_run = False

        validation_steps = 1

        validation_start_time = time.time()
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[
                          step * batch_size:(step + 1) * batch_size
                          ]
                batch_y = y_val[
                          step * batch_size:(step + 1) * batch_size
                          ]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(
                output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        validation_end_time = round(time.time() - validation_start_time, 2)
        if epoch is None:
            print(f'validation, ' +
                  f'acc: {validation_accuracy:.3f}, ' +
                  f'loss: {validation_loss:.3f} ' +
                  f'time: {validation_end_time}s'
                  )
        else:
            print(f'validation - {epoch}, ' +
                  f'acc: {validation_accuracy:.3f}, ' +
                  f'loss: {validation_loss:.3f} ' +
                  f'time: {validation_end_time}s'
                  )

    def predict(self, X, *, batch_size=None):

        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):

            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:

            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def get_parameters(self):

        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):

        for parameter_set, layer in zip(
                parameters,
                self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        with open(f'{path}', 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(f'{path}', 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases', 'next', 'prev']:
                layer.__dict__.pop(property, None)
            if hasattr(layer, 'inputs_reshaped'):
                layer.__dict__.pop('inputs_reshaped', None)

        for layer in model.trainable_layers:
            for property in ['weight_cache', 'weight_momentums', 'bias_cache', 'bias_momentums']:
                layer.__dict__.pop(property, None)
            if hasattr(layer, 'inputs_col'):
                layer.__dict__.pop('inputs_col', None)

        model.trainable_layers = None
        model.first_run = True

        with gzip.open(f'{path}.gz', 'wb') as f:
            pickle.dump(model, f, -1)

    def load(self, path):
        with gzip.open(path, 'rb') as f:
            model = pickle.load(f)
            return model