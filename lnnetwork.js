class NeuralNetwork {
  /**
   * @param {Array} layers [input, hidden..., output]
   * @param {Float} (optional) learning_rate
   */
  constructor(layers, learning_rate = 0.01, load_data = null) {
    this.layers = layers;
    this.learning_rate = learning_rate;

    this.activation = NeuralNetwork.activation_functions.sigmoid;

    this.weights = [];
    this.biases = [];

    for (let l = 0; l < layers.length - 1; l++) {
      this.weights.push([]);
      this.biases.push([]);
      for (let n = 0; n < layers[l + 1]; n++) {
        this.weights[l].push([]);
        for (let c = 0; c < layers[l]; c++) {
          this.weights[l][n].push(Math.random() * 2 - 1);
        }
      }
      for (let n = 0; n < this.weights[l].length; n++) {
        this.biases[l].push(Math.random() * 2 - 1);
      }
    }
  }

  /**
   * Returns output data from trained neural network.
   * @param {Array} input array input data
   * @returns {Array}
   */
  predict(input) {
    let activations = input;

    for (let l = 0; l < this.layers.length - 1; l++) {
      const layer_outputs = [];
      for (let n = 0; n < this.layers[l + 1]; n++) {
        let sum = 0;
        for (let c = 0; c < this.layers[l]; c++) {
          sum += activations[c] * this.weights[l][n][c];
        }
        layer_outputs.push(this.activation.function(sum + this.biases[l][n]));
      }
      activations = layer_outputs;
    }

    return activations;
  }

  /**
   * @param {Array} input array input data
   * @param {Array} target array target output data
   * @param {Number} (optional) epoch epoch num.
   */
  train(input, target, epoch = 1) {
    for (let epoch_index = 0; epoch_index < epoch; epoch_index++) {
      let activations = [input];
      let layer_inputs = [input];

      for (let l = 0; l < this.layers.length - 1; l++) {
        const layer_outputs = [];
        for (let n = 0; n < this.layers[l + 1]; n++) {
          let sum = 0;
          for (let c = 0; c < this.layers[l]; c++) {
            sum += activations[l][c] * this.weights[l][n][c];
          }
          const output = this.activation.function(sum + this.biases[l][n]);
          layer_outputs.push(output);
        }
        activations.push(layer_outputs);
        layer_inputs.push(layer_outputs);
      }

      const errors = [];
      let output_error = [];
      for (let n = 0; n < this.layers[this.layers.length - 1]; n++) {
        output_error.push(target[n] - activations[activations.length - 1][n]);
      }
      errors.push(output_error);

      for (let l = this.layers.length - 2; l > 0; l--) {
        const hidden_error = [];
        for (let c = 0; c < this.layers[l]; c++) {
          let error = 0;
          for (let n = 0; n < this.layers[l + 1]; n++) {
            error += errors[0][n] * this.weights[l][n][c];
          }
          hidden_error.push(error);
        }
        errors.unshift(hidden_error);
      }

      for (let l = 0; l < this.layers.length - 1; l++) {
        for (let n = 0; n < this.layers[l + 1]; n++) {
          for (let c = 0; c < this.layers[l]; c++) {
            this.weights[l][n][c] +=
              this.learning_rate *
              errors[l][n] *
              this.activation.derivative(layer_inputs[l + 1][n]) *
              activations[l][c];
          }
          this.biases[l][n] +=
            this.learning_rate *
            errors[l][n] *
            this.activation.derivative(layer_inputs[l + 1][n]);
        }
      }
    }
  }

  /**
   * Activation functions: sigmoid, linear, relu and tanh
   */
  static activation_functions = {
    sigmoid: {
      function: (x) => {
        return 1 / (1 + Math.exp(-x));
      },
      derivative: (y) => {
        return y * (1 - y);
      },
    },
    linear: {
      function: (x) => {
        return x;
      },
      derivative: (y) => {
        return 1;
      },
    },
    relu: {
      function: (x) => {
        return x >= 0 ? x : 0;
      },
      derivative: (y) => {
        return y > 0 ? 1 : 0;
      },
    },
    tanh: {
      function: (x) => {
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
      },
      derivative: (y) => {
        return (
          1 -
          NeuralNetwork.activation_functions.tanh.function(y) *
            NeuralNetwork.activation_functions.tanh.function(y)
        );
      },
    },
  };

  /**
   * Set activation function.
   * USAGE: set_activation_function(NeuralNetwork.activation_functions.sigmoid)
   */
  set_activation_function(activation) {
    this.activation = activation;
  }

  save() {
    const blob = new Blob([JSON.stringify(this, null, 4)], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "neural_network.txt";
    document.body.appendChild(link);
    link.dispatchEvent(
      new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        view: window,
      })
    );
    document.body.removeChild(link);
  }

  open() {
    const blob = new Blob([JSON.stringify(this, null, 4)], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.target = "_blank";
    document.body.appendChild(link);
    link.dispatchEvent(
      new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        view: window,
      })
    );
    document.body.removeChild(link);
  }
}
