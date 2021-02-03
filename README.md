# Fine
A keras-like neural network framework built purely using Python and Numpy that's just that, fine.

## Table of Contents  
[1- How to use](#how-to-use)  
[2- Demo](#demo)  
[3- Technical Specifications](#technical)

## <a name="how-to-use"></a> How to use
```
git clone git@github.com:haidousm/fine.git
cd fine
python3 -m pip install -r requirements.txt
```

## <a name="demo"></a> Demo
### [MNIST Demo Link](https://haidousm.com/fine-mnist-demo/)
Demo was built using javascript for the frontend and a flask server to serve predictions from model.

Demo model creation & training:

```
import numpy as np

from layers.Layer_Flatten import Layer_Flatten
from layers.Layer_Dense import Layer_Dense

from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax

from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.Model import Model
from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

X, y, X_test, y_test = mnist_data()
X = np.array([image.astype(np.float32) / 255 for image in X])
X_test = np.array([image.astype(np.float32) / 255 for image in X_test])

model = Model()
model.add(Layer_Flatten())
model.add(Layer_Dense(784, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossEntropy(),
          optimizer=Optimizer_SGD(learning_rate=0.1, decay=1e-3, momentum=.9),
          accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y,
            epochs=5,
            batch_size=128,
            print_every=100)
            
model.evaluate(X_test, y_test, batch_size=128)

```

## <a name="technical"></a>Technical Specifications
### Layers
- [X] Dense Layer
- [X] Dropout Layer
- [X] Flatten Layer
- [ ] 2D Convolutional Layer
- [ ] Max Pool Layer

### Activation Functions
- [X] Rectified Linear (ReLU)
- [X] Sigmoid
- [X] Softmax
- [X] Linear

### Loss Functions
- [X] Categorical Cross Entropy
- [X] Binary Cross Entropy
- [X] Mean Squared Error

### Optimizers
- [X] Stochastic Gradient Descent (SGD) with rate decay and momentum
- [ ] Adaptive Moment Estimation (ADAM)
