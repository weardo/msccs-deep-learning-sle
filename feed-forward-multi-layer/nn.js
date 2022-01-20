const Matrix = require("./matrix");

class NeuralNetwork {
    constructor(input_nodes, hidden_nodes, output_nodes) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);

        this.weights_ih.randomize();
        this.weights_ho.randomize();

        this.bias_h = new Matrix(this.hidden_nodes, 1);
        this.bias_o = new Matrix(this.output_nodes, 1);
        this.bias_h.randomize();
        this.bias_o.randomize();
        this.learning_rate = 0.1;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    dsigmoid(y) {
        return y * (1 - y);
    }

    feedForward(input_array) {
        let inputs = Matrix.fromArray(input_array);

        // Generating the hidden outputs
        let hidden = Matrix.multiply(this.weights_ih, inputs)
        hidden.add(this.bias_h);
        hidden.map(this.sigmoid)

        let output = Matrix.multiply(this.weights_ho, hidden);
        output.add(this.bias_o);

        output.map(this.sigmoid);

        return output.toArray();
    }

    train(input_array, target_array) {
        let inputs = Matrix.fromArray(input_array);

        // Generating the hidden outputs
        let hidden = Matrix.multiply(this.weights_ih, inputs)
        hidden.add(this.bias_h);
        hidden.map(this.sigmoid)

        let outputs = Matrix.multiply(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(this.sigmoid);

        // convert array to matrix
        let targets = Matrix.fromArray(target_array);

        // calculate errors;
        // error = targets - outputs
        let output_errors = Matrix.subtract(targets, outputs);

        // calculate gradient
        let gradients = Matrix.map(outputs, this.dsigmoid);
        gradients.multiply(output_errors);
        gradients.multiply(this.learning_rate);

        // calculate deltas
        let hidden_T = Matrix.transpose(hidden);
        let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

        // adjust the weights by deltas
        this.weights_ho.add(weight_ho_deltas);
        // adjust the bias by its deltas (which is just the gradient)
        this.bias_o.add(gradients);

        // calculate hidden layer errors
        let who_t = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.multiply(who_t, output_errors);

        // calculate hidden gradient
        let hidden_gradient = Matrix.map(hidden, this.dsigmoid);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learning_rate);

        // calculate input to hidden deltas
        let inputs_T = Matrix.transpose(inputs);
        let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        this.weights_ih.add(weight_ih_deltas);
        this.bias_h.add(hidden_gradient);

        // outputs.print();
        // targets.print();
        // error.print();

    }
}

if (typeof module !== 'undefined') {
    module.exports = NeuralNetwork;
  }