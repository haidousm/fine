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
Demo was built using javascript for the frontend, and a flask server to serve predictions from the model.

Demo model creation & training:

```
from datasets import load_mnist
from models import Sequential

from layers import Conv2D
from layers import MaxPool2D
from layers import Flatten
from layers import Dense

from activations import ReLU
from activations import Softmax

from loss import CategoricalCrossEntropy

from models.model_utils import Categorical

from optimizers import Adam

X_train, y_train, X_test, y_test = load_mnist()

model = Sequential(
    layers=[
        Conv2D(16, (1, 3, 3)),
        ReLU(),
        Conv2D(16, (16, 3, 3)),
        ReLU(),
        MaxPool2D((2, 2)),

        Conv2D(32, (16, 3, 3)),
        ReLU(),
        Conv2D(32, (32, 3, 3)),
        ReLU(),
        MaxPool2D((2, 2)),

        Flatten(),
        Dense(1568, 64),
        ReLU(),
        Dense(64, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ],
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(decay=1e-3),
    accuracy=Categorical()
)

model.train(X_train, y_train, epochs=5, batch_size=120, print_every=100)
model.evaluate(X_test, y_test, batch_size=120)


```

## <a name="technical"></a>Technical Specifications
### Layers
- [X] Dense Layer
- [X] Dropout Layer
- [X] Flatten Layer
- [X] 2D Convolutional Layer
- [X] Max Pool Layer

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
- [X] Adaptive Moment Estimation (ADAM)
