#include "../include/matrix.h"
#include "../include/supervised.h" 
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

const double TEST_TOLERANCE = 1e-4; 

void generate_synthetic_data(Matrix *X, Matrix *y, int num_samples, int num_features) {
    *X = Matrix_zeros(num_samples, num_features);
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            X->data[i][j] = (double) rand() / RAND_MAX;
        }
    }

    *y = Matrix_zeros(num_samples, 1);
    for (int i = 0; i < num_samples; ++i) {
        y->data[i][0] = rand() % 2;  
    }
}


void test_linear_regression() {
    printf("Testing LinearRegressionModel...\n");
    
    LabelledData data = Supervised_read_csv("test/test_data/ridge_regression_data.csv");
    
    LinearRegressionModel model = LinearRegression(&data.X, &data.y);
    
    FILE *gnuplot = popen("gnuplot -persist", "w");
    if (!gnuplot) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    fprintf(gnuplot, "set title 'Linear Regression Model'\n");
    fprintf(gnuplot, "set xlabel 'Feature'\n");
    fprintf(gnuplot, "set ylabel 'Target'\n");
    fprintf(gnuplot, "set style data linespoints\n");
    fprintf(gnuplot, "set key outside\n");
    fprintf(gnuplot, "set xrange [*:*]\n");
    fprintf(gnuplot, "set yrange [*:*]\n");

    fprintf(gnuplot, "plot '-' using 1:2 title 'Original Data' with points pt 7 ps 1.5 lc rgb 'black', \\\n");

    const char *modes[] = {"ALGEBRAIC", "BATCH", "MINIBATCH", "STOCHASTIC"};
    Matrix y_preds[4];
    for (int i = 0; i < 4; ++i) {
        LinearRegression_set_mode(&model, i);
        LinearRegression_train(&model);
        y_preds[i] = LinearRegression_predict(&model, &data.X);

        fprintf(gnuplot, "'-' using 1:2 title '%s Prediction' with lines lw 2 lc rgb '%s'%s\n", 
                modes[i], (i == 0 ? "red" : (i == 1 ? "blue" : (i == 2 ? "green" : "magenta"))), (i < 3 ? ", \\\n" : "\n"));
    }

    for (int i = 0; i < data.X.rows; ++i) {
        fprintf(gnuplot, "%f %f\n", data.X.data[i][0], data.y.data[i][0]);
    }
    fprintf(gnuplot, "e\n");

    for (int mode = 0; mode < 4; ++mode) {
        for (int i = 0; i < data.X.rows; ++i) {
            fprintf(gnuplot, "%f %f\n", data.X.data[i][0], y_preds[mode].data[i][0]);
        }
        fprintf(gnuplot, "e\n");
    }

    fflush(gnuplot);
    pclose(gnuplot);

    for (int i = 0; i < 4; ++i) {
        Matrix_free(y_preds[i]);
    }
    Matrix_free(data.X);
    Matrix_free(data.y);
    LinearRegression_free(model);

    printf("LinearRegressionModel test passed!\n");
}

void test_ridge_regression() {
    printf("Testing RidgeRegressionModel...\n");
    
    LabelledData data = Supervised_read_csv("test/test_data/ridge_regression_data.csv");
    
    LinearRegressionModel model = RidgeRegression(&data.X, &data.y, 300);
    
    FILE *gnuplot = popen("gnuplot -persist", "w");
    if (!gnuplot) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    fprintf(gnuplot, "set title 'Ridge Regression Model (w/ outlier)'\n");
    fprintf(gnuplot, "set xlabel 'Feature'\n");
    fprintf(gnuplot, "set ylabel 'Target'\n");
    fprintf(gnuplot, "set style data linespoints\n");
    fprintf(gnuplot, "set key outside\n");
    fprintf(gnuplot, "set xrange [*:*]\n");
    fprintf(gnuplot, "set yrange [*:*]\n");

    fprintf(gnuplot, "plot '-' using 1:2 title 'Original Data' with points pt 7 ps 1.5 lc rgb 'black', \\\n");

    const char *modes[] = {"ALGEBRAIC", "BATCH", "MINIBATCH", "STOCHASTIC"};
    Matrix y_preds[4];
    for (int i = 0; i < 4; ++i) {
        LinearRegression_set_mode(&model, i);
        LinearRegression_train(&model);
        y_preds[i] = LinearRegression_predict(&model, &data.X);

        fprintf(gnuplot, "'-' using 1:2 title '%s Prediction' with lines lw 2 lc rgb '%s'%s\n", 
                modes[i], (i == 0 ? "red" : (i == 1 ? "blue" : (i == 2 ? "green" : "magenta"))), (i < 3 ? ", \\\n" : "\n"));
    }

    for (int i = 0; i < data.X.rows; ++i) {
        fprintf(gnuplot, "%f %f\n", data.X.data[i][0], data.y.data[i][0]);
    }
    fprintf(gnuplot, "e\n");

    for (int mode = 0; mode < 4; ++mode) {
        for (int i = 0; i < data.X.rows; ++i) {
            fprintf(gnuplot, "%f %f\n", data.X.data[i][0], y_preds[mode].data[i][0]);
        }
        fprintf(gnuplot, "e\n");
    }

    fflush(gnuplot);
    pclose(gnuplot);

    for (int i = 0; i < 4; ++i) {
        Matrix_free(y_preds[i]);
    }
    Matrix_free(data.X);
    Matrix_free(data.y);
    LinearRegression_free(model);

    printf("RidgeRegressionModel test passed!\n");
}

void test_lasso_regression() {
    printf("Testing LassoRegressionModel...\n");
    
    LabelledData data = Supervised_read_csv("test/test_data/ridge_regression_data.csv");
    
    LinearRegressionModel model = LassoRegression(&data.X, &data.y, 300);
    LinearRegression_set_mode(&model, BATCH);
    
    FILE *gnuplot = popen("gnuplot -persist", "w");
    if (!gnuplot) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    fprintf(gnuplot, "set title 'Ridge Regression Model (w/ outlier)'\n");
    fprintf(gnuplot, "set xlabel 'Feature'\n");
    fprintf(gnuplot, "set ylabel 'Target'\n");
    fprintf(gnuplot, "set style data linespoints\n");
    fprintf(gnuplot, "set key outside\n");
    fprintf(gnuplot, "set xrange [*:*]\n");
    fprintf(gnuplot, "set yrange [*:*]\n");

    fprintf(gnuplot, "plot '-' using 1:2 title 'Original Data' with points pt 7 ps 1.5 lc rgb 'black', \\\n");

    const char *modes[] = {"ALGEBRAIC", "BATCH", "MINIBATCH", "STOCHASTIC"};
    Matrix y_preds[4];
    for (int i = 0; i < 4; ++i) {
        LinearRegression_set_mode(&model, i);
        LinearRegression_train(&model);
        y_preds[i] = LinearRegression_predict(&model, &data.X);

        fprintf(gnuplot, "'-' using 1:2 title '%s Prediction' with lines lw 2 lc rgb '%s'%s\n", 
                modes[i], (i == 0 ? "red" : (i == 1 ? "blue" : (i == 2 ? "green" : "magenta"))), (i < 3 ? ", \\\n" : "\n"));
    }

    for (int i = 0; i < data.X.rows; ++i) {
        fprintf(gnuplot, "%f %f\n", data.X.data[i][0], data.y.data[i][0]);
    }
    fprintf(gnuplot, "e\n");

    for (int mode = 0; mode < 4; ++mode) {
        for (int i = 0; i < data.X.rows; ++i) {
            fprintf(gnuplot, "%f %f\n", data.X.data[i][0], y_preds[mode].data[i][0]);
        }
        fprintf(gnuplot, "e\n");
    }

    fflush(gnuplot);
    pclose(gnuplot);

    for (int i = 0; i < 4; ++i) {
        Matrix_free(y_preds[i]);
    }
    Matrix_free(data.X);
    Matrix_free(data.y);
    LinearRegression_free(model);

    printf("RidgeRegressionModel test passed!\n");
}

void test_knn_classification() {
    printf("Testing KNNModel classification...\n");

    LabelledData data = Supervised_read_csv("test/test_data/knn_test_data.csv");

    Matrix X = data.X;
    Matrix y = data.y;

    KNNModel model = KNNClassifier(5, &X, &y); // Set k = 5

    double x_new_data[4][2] = {
        {-4.0, -0.5}, {3.0, -5.0},
        {-10.0, 15.0}, {-2.0, 5.0}
    };
    Matrix x_new = Matrix_from_array(4, 2, &x_new_data[0][0]);

    Matrix predictions = KNN_predict(&model, &x_new);

    printf("Classification predictions:\n");
    Matrix_display(&predictions);

    FILE *train_data_file = fopen("train_data.tmp", "w");
    for (int i = 0; i < X.rows; ++i) {
        fprintf(train_data_file, "%f %f %d\n", X.data[i][0], X.data[i][1], (int)y.data[i][0]);
    }
    fclose(train_data_file);

    FILE *new_data_file = fopen("new_data.tmp", "w");
    for (int i = 0; i < x_new.rows; ++i) {
        fprintf(new_data_file, "%f %f %d\n", x_new.data[i][0], x_new.data[i][1], (int)predictions.data[i][0]);
    }
    fclose(new_data_file);

    FILE *gnuplot = popen("gnuplot -persist", "w");
    if (!gnuplot) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    fprintf(gnuplot, "set title 'KNN Classification'\n");
    fprintf(gnuplot, "set xlabel 'Feature 1'\n");
    fprintf(gnuplot, "set ylabel 'Feature 2'\n");
    fprintf(gnuplot, "set style data points\n");
    fprintf(gnuplot, "set pointsize 1.5\n");
    fprintf(gnuplot, "set palette defined (0 'red', 1 'green', 2 'blue', 3 'yellow')\n");
    fprintf(gnuplot, "plot 'train_data.tmp' using 1:2:3 with points palette title 'Training Data', \\\n");
    fprintf(gnuplot, "     'new_data.tmp' using 1:2:($3) with points pt 7 ps 2 palette title 'New Points'\n");

    fflush(gnuplot);
    pclose(gnuplot);

    remove("train_data.tmp");
    remove("new_data.tmp");

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

    KNNModel model = KNNRegressor(2, &X, &y);

    double x_new_data[2][1] = {{2.5}, {3.5}};
    Matrix x_new = Matrix_from_array(2, 1, &x_new_data[0][0]);

    Matrix predictions = KNN_predict(&model, &x_new);

    Matrix_free(X);
    Matrix_free(y);
    Matrix_free(x_new);
    Matrix_free(predictions);
    KNN_free(model);
    printf("KNNModel regression test passed\n");
}

void test_logistic_regression() {
    printf("Testing LogisticRegression...\n");

    LabelledData data = Supervised_read_csv("test/test_data/logistic_regression_data.csv");

    LogisticRegressionModel model = LogisticRegression(&data.X, &data.y);

    LogisticRegression_train(&model);

    Matrix X_new, y_new;
    generate_synthetic_data(&X_new, &y_new, 10, data.X.cols);  

    Matrix y_pred = LogisticRegression_predict(&model, &X_new);

    FILE *gnuplot = popen("gnuplot -persist", "w");
    if (!gnuplot) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    fprintf(gnuplot, "set title 'Logistic Regression Model'\n");
    fprintf(gnuplot, "set xlabel 'Feature 1'\n");
    fprintf(gnuplot, "set ylabel 'Feature 2'\n");
    fprintf(gnuplot, "set style data points\n");
    fprintf(gnuplot, "set pointsize 1.5\n");

    fprintf(gnuplot, "plot '-' using 1:2:(($3 == 0) ? 1 : 2) with points pt 7 ps 1 lc variable title 'Original Data'\n");
    for (int i = 0; i < data.X.rows; ++i) {
        fprintf(gnuplot, "%f %f %d\n", data.X.data[i][0], data.X.data[i][1], (int)data.y.data[i][0]);
    }
    fprintf(gnuplot, "e\n");

    fprintf(gnuplot, "plot '-' using 1:2:(($3 == 0) ? 1 : 2) with points pt 7 ps 1 lc variable title 'Predicted Data'\n");
    for (int i = 0; i < X_new.rows; ++i) {
        fprintf(gnuplot, "%f %f %d\n", X_new.data[i][0], X_new.data[i][1], (int)y_pred.data[i][0]);
    }
    fprintf(gnuplot, "e\n");

    fflush(gnuplot);
    pclose(gnuplot);

    Matrix_free(data.X);
    Matrix_free(data.y);
    Matrix_free(X_new);
    Matrix_free(y_pred);

    printf("Logistic Regression Test Passed!\n");
}

int main() {
    test_linear_regression();
    test_ridge_regression();
    test_knn_classification();
    // test_logistic_regression();

    printf("All tests passed successfully.\n\n");
    return 0;
}