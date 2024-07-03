#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

typedef struct {
    double (*apply)(double);
    double (*apply_derivative)(double);
} ActivationFunction;

typedef struct {
    int input_size;
    Matrix weights; 
    double bias; 
    ActivationFunction activation;
} Neuron;

typedef struct {
    int num_neurons;
    Neuron *neurons;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
} NeuralNetwork;

ActivationFunction NN_get_sigmoid();

Neuron NN_create_neuron(int input_size, ActivationFunction activation);
Layer NN_create_layer(int num_neurons, int input_size, ActivationFunction activation);

Matrix NN_forward_pass(NeuralNetwork *network, const Matrix *input);
void NN_backward_pass(NeuralNetwork *network, const Matrix *input);
