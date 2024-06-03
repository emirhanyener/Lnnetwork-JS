class NeuralNetwork {
  /**
   * @param {Array} layers [input, hidden..., output]
   * @param {Float} learning_rate (optional) learning_rate
   * @param {Object} load_data (optional) pretrained neural network data
   */
  constructor(layers, learning_rate = 0.01, load_data = null) {
    this.layers = layers;
    this.learning_rate = learning_rate;

    this.total_epoch = 0;
    this.last_metrics = {};

    this.activation = NeuralNetwork.relu;

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

  visualize(
    canvasId,
    params = {
      backgroundColor: "white",
      neuronFill: "white",
      neuronBorder: "gray",
      size: 1,
    }
  ) {
    console.log(this.weights);
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = params.backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let l = 0; l < this.layers.length; l++) {
      for (let n = 0; n < this.layers[l]; n++) {
        const fromX = (l * canvas.width) / this.layers.length + 50;
        const fromY = (canvas.height / this.layers[l]) * n + 50;

        for (let w = 0; w < this.layers[l + 1]; w++) {
          const toX = ((l + 1) * canvas.width) / this.layers.length + 50;
          const toY = (canvas.height / this.layers[l + 1]) * w + 50;

          ctx.beginPath();
          ctx.strokeStyle = this.weights[l][w][n] > 0 ? "green" : "red";
          ctx.lineWidth = this.weights[l][w][n] * 0.1 * params.size;
          ctx.moveTo(fromX * params.size, fromY * params.size);
          ctx.lineTo(toX * params.size, toY * params.size);
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.lineWidth = 4 * params.size;
        ctx.arc(
          fromX * params.size,
          fromY * params.size,
          20 * params.size,
          0,
          2 * Math.PI
        );
        ctx.strokeStyle = params.neuronBorder;
        ctx.fillStyle = params.neuronFill;
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
        if (l == this.layers.length - 2) {
          layer_outputs.push(
            NeuralNetwork.sigmoid.function(sum + this.biases[l][n])
          );
        } else {
          layer_outputs.push(this.activation.function(sum + this.biases[l][n]));
        }
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
        if (l == this.layers.length - 2) {
          const output = NeuralNetwork.sigmoid.function(
            sum + this.biases[l][n]
          );
          layer_outputs.push(output);
        } else {
          const output = this.activation.function(sum + this.biases[l][n]);
          layer_outputs.push(output);
        }
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
          if (l == this.layers.length - 2) {
            this.weights[l][n][c] +=
              this.learning_rate *
              errors[l][n] *
              NeuralNetwork.sigmoid.derivative(layer_inputs[l + 1][n]) *
              activations[l][c];
          } else {
            this.weights[l][n][c] +=
              this.learning_rate *
              errors[l][n] *
              this.activation.derivative(layer_inputs[l + 1][n]) *
              activations[l][c];
          }
        }
        if (l == this.layers.length - 2) {
          this.biases[l][n] +=
            this.learning_rate *
            errors[l][n] *
            NeuralNetwork.sigmoid.derivative(layer_inputs[l + 1][n]);
        } else {
          this.biases[l][n] +=
            this.learning_rate *
            errors[l][n] *
            this.activation.derivative(layer_inputs[l + 1][n]);
        }
      }
    }
  }

  //Train with inputs, targets and return confusion matrix, accuracy score, precision score, recall score, f1 score
  train_all(inputs, targets, epoch = 1, test_percent = 0.2) {
    const test_data_num = Math.floor(inputs.length * test_percent);

    for (let epoch_index = 0; epoch_index < epoch; epoch_index++) {
      const input_data = [...inputs];
      const target_data = [...targets];
      const test_input_data = [];
      const test_target_data = [];

      for (let index = 0; index < test_data_num; index++) {
        const random_index = Math.floor(Math.random() * input_data.length);
        test_input_data.push(input_data.splice(random_index, 1)[0]);
        test_target_data.push(target_data.splice(random_index, 1)[0]);
      }

      for (let data_index = 0; data_index < input_data.length; data_index++) {
        this.train(input_data[data_index], target_data[data_index]);
      }

      const outputs = this.test(test_input_data, test_target_data);

      this.last_metrics = NeuralNetwork.calculate_metrics(outputs);

      this.total_epoch++;
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
    this.total_epoch = data.total_epoch;
    this.last_metrics = data.last_metrics;
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
    //Column predicted
    //Row actual
    for (let i = 0; i < inputs.length; i++) {
      const _predicted = this.predict(inputs[i]);
      outputs[targets[i].indexOf(1)][
        _predicted.indexOf(Math.max(..._predicted))
      ]++;
    }
    return outputs;
  }

  static calculate_metrics(confusionMatrix) {
    const numClasses = confusionMatrix.length;
    const metrics = {
      accuracy: 0,
      precision: [],
      recall: [],
      f1Score: [],
      weightedPrecision: 0,
      weightedRecall: 0,
      weightedF1Score: 0,
    };

    let totalSamples = 0;
    let totalCorrectPredictions = 0;
    const support = new Array(numClasses).fill(0);

    for (let i = 0; i < numClasses; i++) {
      support[i] = confusionMatrix[i].reduce((a, b) => a + b, 0);
      totalSamples += support[i];
      totalCorrectPredictions += confusionMatrix[i][i];
    }

    metrics.accuracy = totalCorrectPredictions / totalSamples;

    for (let i = 0; i < numClasses; i++) {
      let TP = confusionMatrix[i][i];
      let FN = support[i] - TP;
      let FP = 0;
      let TN = totalSamples - (TP + FN);

      for (let j = 0; j < numClasses; j++) {
        if (j !== i) {
          FP += confusionMatrix[j][i];
          TN -= confusionMatrix[j][i];
        }
      }

      const precision = TP / (TP + FP) || 0;
      const recall = TP / (TP + FN) || 0;
      const f1Score = (2 * (precision * recall)) / (precision + recall) || 0;

      metrics.precision.push(precision);
      metrics.recall.push(recall);
      metrics.f1Score.push(f1Score);

      metrics.weightedPrecision += precision * support[i];
      metrics.weightedRecall += recall * support[i];
      metrics.weightedF1Score += f1Score * support[i];
    }

    metrics.weightedPrecision /= totalSamples;
    metrics.weightedRecall /= totalSamples;
    metrics.weightedF1Score /= totalSamples;

    return {
      confusion_matrix: confusionMatrix,
      accuracy: metrics.accuracy,
      precision: metrics.weightedPrecision,
      recall: metrics.weightedRecall,
      f1score: metrics.weightedF1Score,
    };
  }
}
