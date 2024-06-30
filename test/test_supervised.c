#include "../include/matrix.h"
#include "../include/supervised.h" 
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

const double TEST_TOLERANCE = 1e-4; 

// Macro for writing to GNUplot
#define GP_WRITE(fmt, ...) \
        do { \
            fprintf(gnuplot_file, fmt, ##__VA_ARGS__); \
        } while (0)

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
    
    FILE *gnuplot_file = popen("gnuplot -persist", "w");
    if (!gnuplot_file) {
        fprintf(stderr, "Error opening gnuplot or gnuplot script file\n");
        if (gnuplot_file) pclose(gnuplot_file);
        return;
    }

    const char *gnuplot_commands[] = {
        "set title 'Linear Regression Model'\n",
        "set xlabel 'Feature'\n",
        "set ylabel 'Target'\n",
        "set style data linespoints\n",
        "set key outside\n",
        "set xrange [*:*]\n",
        "set yrange [*:*]\n",
        "plot '-' using 1:2 title 'Original Data' with points pt 7 ps 1.5 lc rgb 'black', \\\n"
    };

    for (int i = 0; i < sizeof(gnuplot_commands) / sizeof(gnuplot_commands[0]); ++i) {
        GP_WRITE("%s", gnuplot_commands[i]);
    }

    const char *modes[] = {"ALGEBRAIC", "BATCH", "MINIBATCH"};
    Matrix y_preds[3];
    for (int i = 0; i < 3; ++i) {
        LinearRegression_set_mode(&model, i);
        LinearRegression_train(&model);
        y_preds[i] = LinearRegression_predict(&model, &data.X);

        GP_WRITE("'-' using 1:2 title '%s Prediction' with lines lw 2 lc rgb '%s'%s", 
                modes[i], (i == 0 ? "red" : (i == 1 ? "blue" : (i == 2 ? "green" : "magenta"))), (i < 3 ? ", \\\n" : "\n"));
    }

    for (int i = 0; i < data.X.rows; ++i) {
        GP_WRITE("%f %f\n", data.X.data[i][0], data.y.data[i][0]);
    }
    GP_WRITE("e\n");

    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < data.X.rows; ++i) {
            GP_WRITE("%f %f\n", data.X.data[i][0], y_preds[mode].data[i][0]);
        }
        GP_WRITE("e\n");
    }

    fflush(gnuplot_file);
    pclose(gnuplot_file);

    for (int i = 0; i < 3; ++i) {
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
    
    FILE *gnuplot_file = popen("gnuplot -persist", "w");
    if (!gnuplot_file) {
        fprintf(stderr, "Error opening gnuplot or gnuplot script file\n");
        if (gnuplot_file) pclose(gnuplot_file);
        return;
    }

    const char *gnuplot_commands[] = {
        "set title 'Ridge Regression Model'\n",
        "set xlabel 'Feature'\n",
        "set ylabel 'Target'\n",
        "set style data linespoints\n",
        "set key outside\n",
        "set xrange [*:*]\n",
        "set yrange [*:*]\n",
        "plot '-' using 1:2 title 'Original Data' with points pt 7 ps 1.5 lc rgb 'black', \\\n"
    };

    for (int i = 0; i < sizeof(gnuplot_commands) / sizeof(gnuplot_commands[0]); ++i) {
        GP_WRITE("%s", gnuplot_commands[i]);
    }

    const char *modes[] = {"ALGEBRAIC", "BATCH", "MINIBATCH"};
    Matrix y_preds[3];
    for (int i = 0; i < 3; ++i) {
        LinearRegression_set_mode(&model, i);
        LinearRegression_train(&model);
        y_preds[i] = LinearRegression_predict(&model, &data.X);

        GP_WRITE("'-' using 1:2 title '%s Prediction' with lines lw 2 lc rgb '%s'%s", 
                modes[i], (i == 0 ? "red" : (i == 1 ? "blue" : (i == 2 ? "green" : "magenta"))), (i < 3 ? ", \\\n" : "\n"));
    }

    for (int i = 0; i < data.X.rows; ++i) {
        GP_WRITE("%f %f\n", data.X.data[i][0], data.y.data[i][0]);
    }
    GP_WRITE("e\n");

    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < data.X.rows; ++i) {
            GP_WRITE("%f %f\n", data.X.data[i][0], y_preds[mode].data[i][0]);
        }
        GP_WRITE("e\n");
    }

    fflush(gnuplot_file);
    pclose(gnuplot_file);

    for (int i = 0; i < 3; ++i) {
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

    FILE *gnuplot_file = popen("gnuplot -persist", "w");
    if (!gnuplot_file) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    GP_WRITE("set title 'KNN Classification'\n");
    GP_WRITE("set xlabel 'Feature 1'\n");
    GP_WRITE("set ylabel 'Feature 2'\n");
    GP_WRITE("set style data points\n");
    GP_WRITE("set pointsize 1.5\n");
    GP_WRITE("set palette defined (0 'red', 1 'green', 2 'blue', 3 'yellow')\n");
    GP_WRITE("plot 'train_data.tmp' using 1:2:3 with points palette title 'Training Data', \\\n");
    GP_WRITE("     'new_data.tmp' using 1:2:($3) with points pt 7 ps 2 palette title 'New Points'\n");

    fflush(gnuplot_file);
    pclose(gnuplot_file);

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

    FILE *gnuplot_file = popen("gnuplot -persist", "w");
    if (!gnuplot_file) {
        fprintf(stderr, "Error opening gnuplot\n");
        fclose(gnuplot_file);
        return;
    }

    GP_WRITE("set title 'Logistic Regression Model'\n");
    GP_WRITE("set xlabel 'Feature 1'\n");
    GP_WRITE("set ylabel 'Feature 2'\n");
    GP_WRITE("set style data points\n");
    GP_WRITE("set pointsize 1.5\n");

    GP_WRITE("plot '-' using 1:2:($3 == 1 ? 1 : 2):($3 == 1 ? 2 : 1) with points pt variable lc variable title 'Original Data',\\\n");
    GP_WRITE("     '-' using 1:2:($3 > 0.5 ? 2 : 1) with points pt 7 lc variable title 'Predicted Data'\n");
    for (int i = 0; i < data.X.rows; ++i) {
        int class = (int)data.y.data[i][0];
        GP_WRITE("%f %f %d\n", data.X.data[i][0], data.X.data[i][1], class);
    }
    GP_WRITE("e\n");

    for (int i = 0; i < X_new.rows; ++i) {
        double prediction = y_pred.data[i][0];
        GP_WRITE("%f %f %f\n", X_new.data[i][0], X_new.data[i][1], prediction);
    }
    GP_WRITE("e\n");

    double b = model.params.data[0][0];
    double w1 = model.params.data[1][0];
    double w2 = model.params.data[2][0];

    GP_WRITE("set xrange [-5:5]\n");
    GP_WRITE("set yrange [-5:5]\n");
    GP_WRITE("set samples 1000\n");
    GP_WRITE("plot 1.0 / (1.0 + exp(-(%.10lf + %.10lf*x + %.10lf*y))) with lines title 'Decision Boundary'\n", b, w1, w2);

    fclose(gnuplot_file);
    fflush(gnuplot_file);
    pclose(gnuplot_file);

    Matrix_free(data.X);
    Matrix_free(data.y);
    Matrix_free(X_new);
    Matrix_free(y_pred);

    printf("Logistic Regression Test Passed!\n");
}


void test_naive_bayes() {
    printf("Testing Gaussian Naive Bayes...\n");

    LabelledData data = Supervised_read_csv("test/test_data/naive_bayes_data.csv");

    GaussianNBCModel model = GaussianNBC(&data.X, &data.y);

    LabelledData new_data = Supervised_read_csv("test/test_data/naive_bayes_new_data.csv");

    Matrix predictions = GaussianNBC_predict(&model, &new_data.X);

    printf("Predictions:\n");
    Matrix_display(&predictions);

    Matrix_free(data.X);
    Matrix_free(data.y);
    Matrix_free(new_data.X);
    Matrix_free(new_data.y);
    Matrix_free(predictions);
    Matrix_free(model.means);
    Matrix_free(model.variances);
    Matrix_free(model.priors);

    printf("Gaussian Naive Bayes test passed!\n");
}

int main() {
    test_linear_regression();
    test_ridge_regression();
    test_knn_classification();
    test_logistic_regression();

    printf("All tests passed successfully.\n\n");
    return 0;
}