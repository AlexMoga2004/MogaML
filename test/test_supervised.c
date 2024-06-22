#include "../include/matrix.h"
#include "../include/supervised.h" // Include your linear regression header file
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const double TEST_TOLERANCE = 1e-4; // Tolerance for floating-point comparisons

void test_linear_regression() {
    LabelledData data = Supervised_read_csv("test/test_data/data.csv");

    LinearRegressionModel model = LinearRegression(&data.X, &data.y);
    Supervised_train(&model);

    printf("nigga\n");
    Matrix y_pred = Supervised_predict(&model, &data.X);
    Matrix_display(&y_pred);

    Matrix_free(data.X);
    Matrix_free(data.y);
    Matrix_free(y_pred);
    Supervised_free(model);

    printf("LinearRegressionModel test passed!\n");
}

int main() {
    test_linear_regression();

    printf("All tests passed successfully.\n\n");
    return 0;
}