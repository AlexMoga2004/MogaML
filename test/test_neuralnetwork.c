// #include "matrix.h"
// #include "neuralnetwork.h"

// void test_forward_propagation() {
//     // Set up a simple network with predefined weights and biases
//     Neuron neurons_layer1[] = {
//         {Matrix_zeros(2, 1), 0.0, SigmoidActivation},
//         {Matrix_zeros(2, 1), 0.0, SigmoidActivation}
//     };
//     neurons_layer1[0].weights.data[0][0] = 0.15;  // weight 1 for neuron 1
//     neurons_layer1[0].weights.data[1][0] = 0.2;   // weight 2 for neuron 1
//     neurons_layer1[1].weights.data[0][0] = 0.25;  // weight 1 for neuron 2
//     neurons_layer1[1].weights.data[1][0] = 0.3;   // weight 2 for neuron 2
//     neurons_layer1[0].bias = 0.35;
//     neurons_layer1[1].bias = 0.35;

//     Neuron neurons_layer2[] = {
//         {Matrix_zeros(2, 1), 0.0, SigmoidActivation}
//     };
//     neurons_layer2[0].weights.data[0][0] = 0.4;   // weight 1 for output neuron
//     neurons_layer2[0].weights.data[1][0] = 0.45;  // weight 2 for output neuron
//     neurons_layer2[0].bias = 0.6;

//     Layer layers[] = {
//         {2, neurons_layer1},
//         {1, neurons_layer2}
//     };

//     NeuralNetwork network = {2, layers};

//     Matrix input = Matrix_zeros(2, 1);
//     input.data[0][0] = 0.05;
//     input.data[1][0] = 0.1;

//     Matrix expected_output = Matrix_zeros(1, 1);
//     expected_output.data[0][0] = 0.75136507;  // Expected output after forward pass

//     Matrix output = NN_forward_pass(&network, &input);

//     assert(fabs(output.data[0][0] - expected_output.data[0][0]) < 1e-6);

//     Matrix_free(input);
//     Matrix_free(expected_output);
//     Matrix_free(output);

//     printf("Forward propagation test passed.\n");
// }
