# Lnnetwork-JS
Lightweight neural network JavaScript library.

## Neural Network
```javascript
// Create neural network. 3 layer, 1 input, 3 hidden, 1 output neurons.
const nn = new NeuralNetwork([1, 3, 1]);
```

## Set Activation Function
```javascript
const nn = new NeuralNetwork([1, 3, 1]);
// Set Activation
nn.set_activation_function(NeuralNetwork.activation_functions.linear)
```

### Activation Functions
- Linear  - linear
- Sigmoid - sigmoid
- ReLU    - relu
- TanH    - tanh

## Training
Train neural network using feed backward.
```javascript
const nn = new NeuralNetwork([1, 3, 1]);
// First parameter input, second parameter target output values
nn.train([0], [0]);
```

## Predict Output
Predict output data using feed forward.
```javascript
const nn = new NeuralNetwork([1, 3, 1]);
// Get output data using input values.
nn.predict([0]);
```

## Save Neural Network Data
Download trained neural network data
```javascript
const nn = new NeuralNetwork([1, 3, 1]);
nn.save();
```

## Open Neural Network Data
Open on new tab trained neural network data
```javascript
const nn = new NeuralNetwork([1, 3, 1]);
nn.open();
```

## Load 
Load using pretrained neural network data
```javascript
//Load using constructor
const nn = new NeuralNetwork([1, 3, 1], load_data = -data-);

// Or load using method
nn.load(-data-);
```

## LICENSE
[MIT](LICENSE)
