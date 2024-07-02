#include "neuralnetwork.h"
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s); 
}

ActivationFunction NN_get_sigmoid() {
    ActivationFunction function;
    function.apply = sigmoid;
    function.apply_derivative = sigmoid_derivative;;
    return function;
}

Neuron NN_create_neuron(int input_size, ActivationFunction activation) {
    Neuron neuron;
    neuron.input_size = input_size;
    neuron.weights = Matrix_zeros(input_size, 1);
    neuron.bias = 0;
    neuron.activation = activation;
    return neuron;
}

Layer NN_create_layer(int num_neurons, int input_size, ActivationFunction activation) {
    Layer layer;
    layer.num_neurons = num_neurons;
    layer.neurons = (Neuron *)malloc(num_neurons * sizeof(Neuron));
    for (int i = 0; i < num_neurons; ++i) {
        layer.neurons[i] = NN_create_neuron(input_size, activation);
    }
    return layer;
}

NeuralNetwork NN_create_network(int *layer_sizes, int num_layers, ActivationFunction activation) {
    NeuralNetwork network;
    network.num_layers = num_layers;
    network.layers = (Layer *)malloc(num_layers * sizeof(Layer));
    
    for (int i = 0; i < num_layers; ++i) {
        network.layers[i] = NN_create_layer(layer_sizes[i], i == 0 ? 0 : layer_sizes[i-1], activation);
    }

    return network;
}

Matrix NN_forward_pass(NeuralNetwork *network, const Matrix *input) {
    Matrix previous_output = *input;

    for (int layer_num = 0; layer_num < network->num_layers; ++layer_num) {
        Layer current_layer = network->layers[layer_num];

        // Compute actiavtion
        Matrix activation = Matrix_zeros(current_layer.num_neurons, 1);
        for (int neuron_num = 0; neuron_num < current_layer.num_neurons; ++neuron_num) {
            Neuron neuron = current_layer.neurons[neuron_num];
            double preactivation = neuron.bias;

            for (int i = 0; i < neuron.weights.rows; ++i) {
                preactivation += neuron.weights.data[i][0] * previous_output.data[i][0];
            }
            activation.data[neuron_num][0] = neuron.activation.apply(preactivation);
        }
            
        Matrix_free(previous_output);
        previous_output = Matrix_clone(&activation);
        Matrix_free(activation);

        if (layer_num == network->num_layers - 1) {
            return previous_output;
        }
    }

    fprintf(stderr, "Error in NN_forward_pass, network has no layers!");
    exit(EXIT_FAILURE);
}

// Matrix NN_backward_pass(NeuralNetwork *network, const Matrix *input) {

// }
