class CNN {
  constructor(featureSize) {
    this.kernel = [];
    this.featureSize = featureSize;
  }

  createRandomKernel(kernelCount, kernelSize) {
    for (let k = 0; k < this.featureSize * kernelCount; k++) {
      this.kernel.push([]);
      for (let y = 0; y < kernelSize; y++) {
        this.kernel[k].push([]);
        for (let x = 0; x < kernelSize; x++) {
          this.kernel[k][y].push(Math.random() - 0.5);
        }
      }
    }
  }

  convolve(inputs, kernelSize = 3, applyRelu = true) {
    const output = [];
    this.createRandomKernel(inputs.length, kernelSize);

    for (let k = 0; k < this.featureSize * inputs.length; k++) {
      output.push([]);
      for (
        let y = 0;
        y < inputs[0].length - (this.kernel[0][0].length - 1);
        y++
      ) {
        output[k].push([]);
        for (
          let x = 0;
          x < inputs[0][0].length - (this.kernel[0][0].length - 1);
          x++
        ) {
          output[k][y].push(0);
        }
      }
    }

    for (let k = 0; k < this.featureSize * inputs.length; k++) {
      for (let y = 0; y < output[k].length; y++) {
        for (let x = 0; x < output[k][0].length; x++) {
          let kernelOutput = 0;
          for (let ky = 0; ky < this.kernel[0].length; ky++) {
            for (let kx = 0; kx < this.kernel[0].length; kx++) {
              for (let i = 0; i < inputs.length; i++) {
                kernelOutput +=
                  inputs[i][y + ky][x + kx] * this.kernel[k][ky][kx];
              }
            }
            if (applyRelu) {
              output[k][y][x] = Math.max(0, kernelOutput);
            } else {
              output[k][y][x] = kernelOutput;
            }
          }
        }
      }
    }
    return output;
  }

  static pool(inputs, kernelSize) {
    const output = [];
    for (let k = 0; k < inputs.length; k++) {
      output.push([]);
      for (let y = 0; y < inputs[0].length / kernelSize; y++) {
        output[k].push([]);
        for (let x = 0; x < inputs[0][0].length / kernelSize; x++) {
          output[k][y].push(0);
        }
      }
    }

    for (let k = 0; k < inputs.length; k++) {
      for (let y = 0; y < output[k].length; y++) {
        for (let x = 0; x < output[k][0].length; x++) {
          let kernelOutput = 0;
          for (let ky = 0; ky < kernelSize; ky++) {
            for (let kx = 0; kx < kernelSize; kx++) {
              if((y * kernelSize) + ky < inputs[k].length && (x * kernelSize) + kx < inputs[k][0].length){
                kernelOutput = Math.max(
                  inputs[k][(y * kernelSize) + ky][(x * kernelSize) + kx], kernelOutput);
              }
            }
            output[k][y][x] = kernelOutput;
          }
        }
      }
    }

    return output;
  }

  static flat(inputs){
    const output = [];
    for (let k = 0; k < inputs.length; k++) {
      for (let y = 0; y < inputs[0].length; y++) {
        for (let x = 0; x < inputs[0][0].length; x++) {
          output.push(inputs[k][y][x]);
        }
      }
    }
    return output;
  }
}
