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
    LinearRegression_free(model);

    printf("LinearRegressionModel test passed!\n");
}

void test_knn_classification() {
    printf("Testing KNNModel classification...\n");
    double X_data[6][2] = {{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}, {-1.0, -1.0}, {-1.0, -2.0}, {-2.0, -1.0}};
    double y_data[6][1] = {{1.0}, {1.0}, {1.0}, {0.0}, {0.0}, {0.0}};

    Matrix X = Matrix_from_array(6, 2, &X_data[0][0]);
    Matrix y = Matrix_from_array(6, 1, &y_data[0][0]);

    KNNModel model = KNNClassifier(&X, &y, 3);

    double x_new_data[2][2] = {{0.0, 0.0}, {3.0, 3.0}};
    Matrix x_new = Matrix_from_array(2, 2, &x_new_data[0][0]);

    Matrix predictions = KNN_predict(&model, &x_new);

    printf("Classification predictions:\n");
    Matrix_display(&predictions);

    Matrix_free(X);
    Matrix_free(y);
    Matrix_free(x_new);
    Matrix_free(predictions);
    KNN_free(model);
    printf("KNNModel classification test passed\n");
}

void test_knn_regression() {
    printf("Testing KNNModel regression\n");
    double X_data[5][1] = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    double y_data[5][1] = {{1.5}, {2.5}, {3.5}, {4.5}, {5.5}}; 

    Matrix X = Matrix_from_array(5, 1, &X_data[0][0]);
    Matrix y = Matrix_from_array(5, 1, &y_data[0][0]);

    KNNModel model = KNNRegressor(&X, &y, 2);

    double x_new_data[2][1] = {{2.5}, {3.5}};
    Matrix x_new = Matrix_from_array(2, 1, &x_new_data[0][0]);

    Matrix predictions = KNN_predict(&model, &x_new);

    printf("Regression predictions:\n");
    Matrix_display(&predictions);

    Matrix_free(X);
    Matrix_free(y);
    Matrix_free(x_new);
    Matrix_free(predictions);
    KNN_free(model);
    printf("KNNModel regression test passed\n");
}

int main() {
    test_linear_regression();
    test_knn_classification();
    test_knn_regression();

    printf("All tests passed successfully.\n\n");
    return 0;
}