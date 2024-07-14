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

void NN_backward_pass(NeuralNetwork *network, const Matrix *input, const Matrix *expected_output) {
    int num_layers = network->num_layers;
    Matrix *activations = (Matrix*)malloc((num_layers + 1) * sizeof(Matrix));
    Matrix *z_values = (Matrix*)malloc((num_layers) * sizeof(Matrix));

    if (activations == NULL || z_values == NULL) {
        fprintf(stderr, "Error in NN_backward_pass, failed to initialise activations and/or z_values");
        exit(EXIT_FAILURE); 
    }
    
    activations[0] = Matrix_clone(input);
    for (int layer_num = 0; layer_num < num_layers; ++layer_num) {
        Layer current_layer = network->layers[layer_num];
        Matrix z = Matrix_zeros(current_layer.num_neurons, 1);
        Matrix activation = Matrix_zeros(current_layer.num_neurons, 1);

        for (int neuron_num = 0; neuron_num < current_layer.num_neurons; ++neuron_num) {
            Neuron neuron = current_layer.neurons[neuron_num];
            double preactivation = neuron.bias;

            for (int i = 0; i < neuron.weights.rows; ++i) {
                preactivation += neuron.weights.data[i][0] * activations[layer_num].data[i][0];
            }
            z.data[neuron_num][0] = preactivation;
            activation.data[neuron_num][0] = neuron.activation.apply(preactivation);
        }

        z_values[layer_num] = z;
        activations[layer_num + 1] = activation;
    }

    Matrix output_error = Matrix_sub(&activations[num_layers], expected_output);  

    for (int layer_num = num_layers - 1; layer_num >= 0; --layer_num) {
        Layer current_layer = network->layers[layer_num];
        Matrix layer_error = Matrix_zeros(current_layer.num_neurons, 1);

        for (int neuron_num = 0; neuron_num < current_layer.num_neurons; ++neuron_num) {
            Neuron *neuron = &current_layer.neurons[neuron_num];
            double z = z_values[layer_num].data[neuron_num][0];
            double delta = output_error.data[neuron_num][0] * neuron->activation.apply_derivative(z);

            layer_error.data[neuron_num][0] = delta;

            for (int i = 0; i < neuron->weights.rows; ++i) {
                neuron->weights.data[i][0] -= 0.01 * delta * activations[layer_num].data[i][0];
            }
            neuron->bias -= 0.01 * delta;
        }

        if (layer_num > 0) {
            Matrix new_error = Matrix_zeros(activations[layer_num].rows, 1);
            for (int i = 0; i < activations[layer_num].rows; ++i) {
                double sum = 0.0;
                for (int neuron_num = 0; neuron_num < current_layer.num_neurons; ++neuron_num) {
                    Neuron neuron = current_layer.neurons[neuron_num];
                    sum += neuron.weights.data[i][0] * layer_error.data[neuron_num][0];
                }
                new_error.data[i][0] = sum;
            }
            Matrix_free(output_error);
            output_error = new_error;
        }

        Matrix_free(layer_error);
    }

    for (int i = 0; i < num_layers + 1; ++i) {
        Matrix_free(activations[i]);
        Matrix_free(z_values[i]);
    }

    free(activations);
    free(z_values);

    Matrix_free(output_error);
}
