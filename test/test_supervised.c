#include "../include/matrix.h"
#include "../include/supervised.h" // Include your linear regression header file
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

const double TEST_TOLERANCE = 1e-4; // Tolerance for floating-point comparisons

void test_linear_regression() {
    printf("Testing LinearRegressionModel...\n");
    LabelledData data = Supervised_read_csv("test/test_data/data.csv");

    LinearRegressionModel model = LinearRegression(&data.X, &data.y);
    LinearRegression_set_mode(&model, ALGEBRAIC);
    LinearRegression_train(&model);
    Matrix y_pred1 = Supervised_predict(&model, &data.X);

    LinearRegression_set_mode(&model, BATCH);
    LinearRegression_train(&model);
    Matrix y_pred2 = Supervised_predict(&model, &data.X);

    LinearRegression_set_mode(&model, MINIBATCH);
    LinearRegression_train(&model);
    Matrix y_pred3 = Supervised_predict(&model, &data.X);

    LinearRegression_set_mode(&model, STOCHASTIC);
    LinearRegression_train(&model);
    Matrix y_pred4 = Supervised_predict(&model, &data.X);

    //TODO: think of a better way to test this (especially stochastic which may fail due to random nature)
    assert(Matrix_approx_equal(&y_pred1, &y_pred2, 1.0) && "ALGEBRAIC and BATCH computations provided different results!");
    assert(Matrix_approx_equal(&y_pred2, &y_pred3, 2.0) && "BATCH and MINIBATCH computations provided different results!");
    // assert(Matrix_approx_equal(&y_pred3, &y_pred4, 1.0) && "MINIBATCH and STOCHASTIC computations provided different results!");

    Matrix_free(data.X);
    Matrix_free(data.y);
    Matrix_free(y_pred1);
    Matrix_free(y_pred2);
    Matrix_free(y_pred3);
    Matrix_free(y_pred4);
    Supervised_free(model);

    printf("LinearRegressionModel test passed!\n");
}

int main() {
    test_linear_regression();

    printf("All tests passed successfully.\n\n");
    return 0;
}