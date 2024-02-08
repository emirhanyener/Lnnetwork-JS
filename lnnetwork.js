class NeuralNetwork {
  /**
   * @param {Array} layers [input, hidden..., output]
   * @param {Float} learning_rate (optional) learning_rate
   * @param {Object} load_data (optional) pretrained neural network data
   */
  constructor(layers, learning_rate = 0.01, load_data = null) {
    this.layers = layers;
    this.learning_rate = learning_rate;

    this.activation = NeuralNetwork.sigmoid;

    //weights and biases
    this.weights = [];
    this.biases = [];

    if (load_data == null) {
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
    } else {
      this.layers = load_data.layers;
      this.learning_rate = load_data.learning_rate;
      this.weights = load_data.weights;
      this.biases = load_data.biases;
    }
  }

  visualize(canvasId) {
    console.log(this.weights);
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");

    // Temizle
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let l = 0; l < this.layers.length; l++) {
      for (let n = 0; n < this.layers[l]; n++) {
        const fromX = (l * canvas.width) / this.layers.length + 50;
        const fromY = (canvas.height / this.layers[l]) * n + 50;

        for (let w = 0; w < this.layers[l + 1]; w++) {
          const toX = ((l + 1) * canvas.width) / this.layers.length + 50;
          const toY = (canvas.height / this.layers[l + 1]) * w + 50;

          ctx.beginPath();
          ctx.strokeStyle = this.weights[l][w][n] > 0 ? "green" : "red";
          ctx.lineWidth = this.weights[l][w][n] * 5;
          ctx.moveTo(fromX, fromY);
          ctx.lineTo(toX, toY);
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.lineWidth = 4;
        ctx.arc(fromX, fromY, 20, 0, 2 * Math.PI);
        ctx.strokeStyle = "gray";
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  /**
   * Returns output data from trained neural network.
   * @param {Array} input array input data
   * @returns {Array} output
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
   * @param {Number} epoch (optional) epoch epoch num.
   */
  train(input, target) {
    for (let index = 0; index < input.length; index++) {
      if (input[index] == NaN) {
        console.error("NaN input");
        console.error("*finishing");
        return;
      } else if (input[index] == null) {
        console.error("null input");
        console.error("*finishing");
        return;
      }
    }
    for (let index = 0; index < target.length; index++) {
      if (target[index] == NaN) {
        console.error("NaN target");
        console.error("*finishing");
        return;
      } else if (target[index] == null) {
        console.error("null target");
        console.error("*finishing");
        return;
      }
    }
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

  train_all(inputs, targets, epoch, test_percent = 0.2) {
    if (inputs.length != this.layers[0]) {
      console.error("input length not equal to input layer neuron count");
      console.error("*finishing");
      return;
    }
    if (targets.length != this.layers[this.layers.length - 1]) {
      console.error(
        "target output length not equal to output layer neuron count"
      );
      console.error("*finishing");
      return;
    }

    const train_data_num = Math.floor(inputs.length * (1 - test_percent));
    for (let epoch_index = 0; epoch_index < epoch; epoch_index++) {
      for (let data_index = 0; data_index < train_data_num; data_index++) {
        this.train(inputs[data_index], targets[data_index]);
      }
    }

    if (Math.floor(inputs.length * test_percent) > 0) {
      const outputs = this.test(
        inputs.slice(train_data_num),
        targets.slice(train_data_num)
      );
      console.log(epoch, "epoch train completed!");
      console.log("accuration:", NeuralNetwork.calculate_accuration(outputs));
    }
  }

  train_all_dataset(dataset, epoch, test_percent = 0.2) {
    const train_data_num = Math.floor(dataset.length * (1 - test_percent));

    for (let epoch_index = 0; epoch_index < epoch; epoch_index++) {
      for (let data_index = 0; data_index < train_data_num; data_index++) {
        this.train(dataset[data_index][0], dataset[data_index][1]);
      }
    }

    const inputs = [];
    for (
      let data_index = train_data_num;
      data_index < dataset.length;
      data_index++
    ) {
      inputs.push(dataset[data_index][0]);
    }
    const targets = [];
    for (
      let data_index = train_data_num;
      data_index < dataset.length;
      data_index++
    ) {
      targets.push(dataset[data_index][1]);
    }

    if (inputs.length > 0) {
      const outputs = this.test(inputs, targets);
      console.log(epoch, "epoch train completed!");
      console.log(
        "accuration:",
        NeuralNetwork.calculate_accuration(outputs, inputs.length)
      );
    }
  }

  /**
   * Activation functions: sigmoid, linear, relu and tanh
   */
  static sigmoid = {
    function: (x) => {
      return 1 / (1 + Math.exp(-x));
    },
    derivative: (y) => {
      return y * (1 - y);
    },
  };
  static linear = {
    function: (x) => {
      return x;
    },
    derivative: (y) => {
      return 1;
    },
  };
  static relu = {
    function: (x) => {
      return x >= 0 ? x : 0;
    },
    derivative: (y) => {
      return y > 0 ? 1 : 0;
    },
  };
  static tanh = {
    function: (x) => {
      return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    },
    derivative: (y) => {
      return (
        1 - NeuralNetwork.tanh.function(y) * NeuralNetwork.tanh.function(y)
      );
    },
  };

  /**
   * Set activation function.
   * USAGE: set_activation_function(NeuralNetwork.activation_functions.sigmoid)
   */
  set_activation_function(activation) {
    this.activation = activation;
  }

  /**
   * Download trained neural network data
   */
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

  /**
   * Open with new tab trained neural network data
   */
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

  /**
   * Load pretrained neural network
   */
  load(data) {
    this.layers = data.layers;
    this.learning_rate = data.learning_rate;
    this.weights = data.weights;
    this.biases = data.biases;
  }

  /**
   *
   * @param {Array} test_data
   * @returns sum_true
   */
  test(inputs, targets) {
    const outputs = [];
    for (let y = 0; y < this.layers[this.layers.length - 1]; y++) {
      outputs.push([]);
      for (let x = 0; x < this.layers[this.layers.length - 1]; x++) {
        outputs[y].push(0);
      }
    }
    for (let i = 0; i < inputs.length; i++) {
      const _predicted = this.predict(inputs[i]);
      outputs[targets[i].indexOf(1)][_predicted.indexOf(Math.max(..._predicted))]++;
    }
    console.log(outputs);
    return outputs;
  }

  static calculate_accuration(output, test_length) {
    let accuration = 0;
    for (let index = 0; index < output.length; index++) {
      accuration += output[index] / test_length;
    }
    return accuration;
  }
}
