#include "supervised.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

LinearRegressionModel LinearRegression(const Matrix *X, const Matrix *y) {
    if (X->rows != y->rows || y->cols != 1) {
        fprintf(stderr, "Error in LinearRegression, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(X->rows, X->cols+1);
    for (int i = 0; i < X->rows; ++i) {
        XPadded.data[i][0] = 1;
    }

    for (int i = 0; i < X->rows; ++i) {
        for (int j = 0; j < X->cols; ++j) {
            XPadded.data[i][j+1] = X->data[i][j];
        }
    }

    LinearRegressionModel model;
    model.data.X = XPadded;
    model.data.y = Matrix_clone(y);
    model.mode = ALGEBRAIC;
    model.params = Matrix_zeros(XPadded.cols, 1);
    model.hyper_params = Matrix_zeros(1, 1);

    model.loss_function = LinearRegression_default_loss();

    return model;
}

LinearRegressionModel RidgeRegression(const Matrix *X, const Matrix *y, double lambda) {
    if (lambda <= 0) {
        fprintf(stderr, "Error in RidgeRegression, λ ≤ 0!");
        exit(EXIT_FAILURE);
    }

    LinearRegressionModel model = LinearRegression(X, y);
    model.loss_function = RidgeRegression_default_loss();

    model.hyper_params = Matrix_zeros(1,1);
    model.hyper_params.data[0][0] = lambda;

    return model;
}

LinearRegressionModel LassoRegression(const Matrix *X, const Matrix *y, double lambda) {
    if (lambda <= 0) {
        fprintf(stderr, "Error in LassoRegression, λ ≤ 0!");
        exit(EXIT_FAILURE);
    }

    LinearRegressionModel model = LinearRegression(X, y);
    model.loss_function = LassoRegression_default_loss();

    model.hyper_params = Matrix_zeros(1,1);
    model.hyper_params.data[0][0] = lambda;

    return model;
}

LossFunction LinearRegression_default_loss() {
    LossFunction loss_function; 
    loss_function.exact_optimum = LinearRegression_exact_optimum;
    loss_function.loss = LinearRegression_compute_mse;
    loss_function.grad = LinearRegression_compute_gradient;

    return loss_function;
}

LossFunction RidgeRegression_default_loss() {
    LossFunction loss_function; 
    loss_function.exact_optimum = LinearRegression_exact_optimum;
    loss_function.loss = LinearRegression_compute_mse;
    loss_function.grad = LinearRegression_compute_gradient;

    return loss_function;
}

LossFunction LassoRegression_default_loss() {
    LossFunction loss_function;
    loss_function.exact_optimum = LassoRegression_exact_optimum;
    loss_function.loss = LassoRegression_compute_mse;
    loss_function.grad = LassoRegression_compute_gradient;

    return loss_function;
}

void LinearRegression_set_loss(LinearRegressionModel *model, LossFunction loss_function) {
    model->loss_function = loss_function;
}

void LinearRegression_train(LinearRegressionModel *model) {
    if (model->mode == ALGEBRAIC) {
        // May throw exception if loss_function has no closed form exact solution
        model->params = model->loss_function.exact_optimum(&model->data.X, &model->data.y, &model->hyper_params);
    } else if (model->mode == BATCH) {
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            Matrix gradients = model->loss_function.grad(&model->data.X, &model->data.y, &model->params, &model->hyper_params);
            Matrix update = Matrix_scale(-LEARNING_RATE, &gradients);
            Matrix new_params = Matrix_add(&model->params, &update);

            Matrix_free(model->params);
            model->params = new_params;
            Matrix_free(gradients);

            if (Matrix_norm(&update) < TOLERANCE) {
                Matrix_free(update);
                break;
            }
            Matrix_free(update);
        }
    } else if (model->mode == MINIBATCH) {
        // Mini-batch Gradient Descent
        int num_batches = model->data.X.rows / MINIBATCH_SIZE;
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            for (int batch = 0; batch < num_batches; ++batch) {
                int start = batch * MINIBATCH_SIZE;
                int end = start + MINIBATCH_SIZE;

                // Create mini-batch
                Matrix X_batch = Matrix_slice_rows(&model->data.X, start, end);
                Matrix y_batch = Matrix_slice_rows(&model->data.y, start, end);

                // Compute gradients for mini-batch
                LinearRegressionModel mini_model = *model;
                mini_model.data.X = X_batch;
                mini_model.data.y = y_batch;
                Matrix gradients = model->loss_function.grad(&mini_model.data.X, &mini_model.data.y, &mini_model.params, &mini_model.hyper_params);

                // Update parameters
                Matrix update = Matrix_scale(-LEARNING_RATE, &gradients);
                Matrix new_params = Matrix_add(&model->params, &update);

                Matrix_free(model->params);
                model->params = new_params;

                Matrix_free(X_batch);
                Matrix_free(y_batch);
                Matrix_free(gradients);
                Matrix_free(update);
            }
        }
    } else if (model->mode == STOCHASTIC) {
        // Stochastic Gradient Descent
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            for (int i = 0; i < model->data.X.rows; ++i) {
                Matrix Xi = Matrix_row(&model->data.X, i);
                Matrix yi = Matrix_row(&model->data.y, i);
                Matrix prediction = Matrix_multiply(&Xi, &model->params);
                Matrix error = Matrix_sub(&prediction, &yi);
                Matrix gradient = Matrix_scale(2.0, &error);
                Matrix gradient_update = Matrix_scale(LEARNING_RATE, &gradient);

                Matrix_free(prediction);
                Matrix_free(error);
                Matrix_free(gradient);

                Matrix XiT = Matrix_transpose(&Xi);
                Matrix delta = Matrix_multiply(&XiT, &gradient_update);

                Matrix new_params = Matrix_sub(&model->params, &delta);

                Matrix_free(model->params);
                model->params = new_params;

                Matrix_free(Xi);
                Matrix_free(yi);
                Matrix_free(XiT);
                Matrix_free(delta);
                Matrix_free(gradient_update);
            }
        }
    } else {
        fprintf(stderr, "Error in LinearRegression_train, undefined computation mode!\n");
        exit(EXIT_FAILURE);
    }

    model->trained = true;
}

void LinearRegression_set_mode(LinearRegressionModel *model, enum ComputationMode mode){
    model->mode = mode;
}

void LinearRegression_free(LinearRegressionModel model) {
    Matrix_free(model.data.X);
    Matrix_free(model.data.y);
    Matrix_free(model.params);
}

// Function to count the number of columns in the CSV file
int count_columns(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char buffer[1024];
    fgets(buffer, sizeof(buffer), file);
    fclose(file);

    int count = 0;
    char *token = strtok(buffer, ",");
    while (token) {
        count++;
        token = strtok(NULL, ",");
    }

    return count;
}

Matrix LinearRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params) {
    Matrix XT = Matrix_transpose(X);
    Matrix XTX = Matrix_multiply(&XT, X);
    Matrix XTXI = Matrix_inverse(&XTX);
    Matrix hat = Matrix_multiply(&XTXI, &XT);
    Matrix result = Matrix_multiply(&hat, y);

    Matrix_free(XT);
    Matrix_free(XTX);
    Matrix_free(XTXI);
    Matrix_free(hat);
    return result;
}

Matrix RidgeRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params) {
    if (hyper_params->rows != 1 && hyper_params->cols != 1) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient, hyper_param must contain single value!");
        exit(EXIT_FAILURE);
    }

    if (hyper_params->data[0][0] <= 0) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient, lambda must be positive");
        exit(EXIT_FAILURE);
    }

    Matrix XT = Matrix_transpose(X);
    Matrix XTX = Matrix_multiply(&XT, X);

    Matrix I = Matrix_identity(XTX.rows);
    I = Matrix_scale(hyper_params->data[0][0], &I);
    XTX = Matrix_add(&XTX, &I);

    Matrix XTXI = Matrix_inverse(&XTX);
    Matrix hat = Matrix_multiply(&XTXI, &XT);
    Matrix result = Matrix_multiply(&hat, y);

    Matrix_free(XT);
    Matrix_free(XTX);
    Matrix_free(XTXI);
    Matrix_free(hat);
    return result;
}

Matrix LassoRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params) {
    fprintf(stderr, "Error in LassoRegression_exact_optimum, closed form solution does not exist! (HINT: don't use ALGEBRAIC computation mode with Lasso)");
    exit(EXIT_FAILURE);
}

double LinearRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params) {
    Matrix predictions = Matrix_multiply(X, params);
    if (predictions.rows != y->rows || predictions.cols != y->cols) {
        fprintf(stderr, "Error in Supervised_compute_mse, dimension mismatch!");
        exit(EXIT_FAILURE);
    }
    double mse = 0.0;

    for (int i = 0; i < predictions.rows; ++i) {
        mse += pow((predictions.data[i][0] - y->data[i][0]), 2);
    }

    Matrix_free(predictions);
    return mse / (double) y->rows;
}

double RidgeRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params) {
    if (hyper_params->rows != 1 && hyper_params->cols != 1) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient, hyper_param must contain single value!");
        exit(EXIT_FAILURE);
    }

    if (hyper_params->data[0][0] <= 0) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient, lambda must be positive");
        exit(EXIT_FAILURE);
    }

    double mse = LinearRegression_compute_mse(X, y, params, hyper_params);
    mse += (hyper_params->data[0][0] * Vector_norm(params, 2)) / (double) y->rows;

    return mse;
}

double LassoRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params) {
    if (hyper_params->rows != 1 && hyper_params->cols != 1) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient, hyper_param must contain single value!");
        exit(EXIT_FAILURE);
    }

    if (hyper_params->data[0][0] <= 0) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient, lambda must be positive");
        exit(EXIT_FAILURE);
    }

    double mse = LinearRegression_compute_mse(X, y, params, hyper_params);
    mse += (hyper_params->data[0][0] * Vector_norm(params, 1)) / (double) y->rows;

    return mse;
}

// Function to read CSV and create matrices
LabelledData Supervised_read_csv(const char *filename) {
    LabelledData data;

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Count the number of lines and columns
    int num_lines = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            num_lines++;
        }
    }
    rewind(file);

    int num_columns = count_columns(filename);
    int num_features = num_columns - 1;

    // Allocate matrices
    data.X = Matrix_zeros(num_lines, num_features);
    data.y = Matrix_zeros(num_lines, 1);

    // Read the data
    int i = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        char *token = strtok(buffer, ",");
        int j = 0;
        while (token != NULL && j < num_features) {
            (data.X).data[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        if (token != NULL) {
            (data.y).data[i][0] = atof(token);
        }
        i++;
    }

    fclose(file);
    return data;
}

Matrix Supervised_predict(const LinearRegressionModel *model, const Matrix *x_new) {
    if (!model->trained) {
        fprintf(stderr, "Error in Supervised_predict, model not trained!");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(x_new->rows, x_new->cols+1);
    for (int i = 0; i < x_new->rows; ++i) {
        XPadded.data[i][0] = 1;
    }

    for (int i = 0; i < x_new->rows; ++i) {
        for (int j = 0; j < x_new->cols; ++j) {
            XPadded.data[i][j+1] = x_new->data[i][j];
        }
    }

    return Matrix_multiply(&XPadded, &model->params);
}

Matrix LinearRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params) {
    Matrix predictions = Matrix_multiply(X, params);
    Matrix errors = Matrix_sub(&predictions, y);
    Matrix XT = Matrix_transpose(X);
    Matrix gradient = Matrix_multiply(&XT, &errors);
    gradient = Matrix_scale(1.0 / y->rows, &gradient);

    Matrix_free(predictions);
    Matrix_free(errors);
    Matrix_free(XT);

    return gradient;
}

Matrix RidgeRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params) {
    if (hyper_params->rows != 1 || hyper_params->cols != 1) {
        fprintf(stderr, "Error in RidgeRegression_compute_gradient: hyper_params must contain a single value (lambda)!");
        exit(EXIT_FAILURE);
    }

    double lambda = hyper_params->data[0][0];

    Matrix predictions = Matrix_multiply(X, params);
    Matrix errors = Matrix_sub(&predictions, y);

    Matrix XT = Matrix_transpose(X);
    Matrix gradient = Matrix_multiply(&XT, &errors);
    gradient = Matrix_scale(1.0 / y->rows, &gradient);

    int num_params = params->rows * params->cols;
    for (int i = 0; i < num_params; ++i) {
        gradient.data[i][0] += lambda * params->data[i][0] / y->rows;
    }

    Matrix_free(predictions);
    Matrix_free(errors);
    Matrix_free(XT);

    return gradient;
}

Matrix LassoRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params) {
    if (hyper_params->rows != 1 || hyper_params->cols != 1) {
        fprintf(stderr, "Error in LassoRegression_compute_gradient: hyper_params must contain a single value (lambda)!");
        exit(EXIT_FAILURE);
    }

    double lambda = hyper_params->data[0][0];

    Matrix predictions = Matrix_multiply(X, params);
    Matrix errors = Matrix_sub(&predictions, y);

    Matrix XT = Matrix_transpose(X);
    Matrix gradient = Matrix_multiply(&XT, &errors);
    gradient = Matrix_scale(1.0 / y->rows, &gradient);

    int num_params = params->rows * params->cols;
    for (int i = 0; i < num_params; ++i) {
        double sign = (params->data[i][0] > 0) ? 1.0 : (params->data[i][0] < 0) ? -1.0 : 0.0;  // subgradient
        gradient.data[i][0] += lambda * sign / y->rows;
    }

    Matrix_free(predictions);
    Matrix_free(errors);
    Matrix_free(XT);

    return gradient;
}

// Helper function to calculate Euclidean distance between two vectors
double euclidean_distance(const double *vec1, const double *vec2, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Comparator for qsort to sort distances
int compare_distances(const void *a, const void *b) {
    double diff = ((double *)a)[0] - ((double *)b)[0];
    return (diff > 0) - (diff < 0);
}

// Function to find the mode of an array (for classification)
double find_mode(const double *array, int size) {
    int *counts = calloc(size, sizeof(int));
    if (counts == NULL) {
        fprintf(stderr, "Memory allocation error in find_mode.\n");
        exit(EXIT_FAILURE);
    }

    double mode = array[0];
    int max_count = 1;

    for (int i = 0; i < size; ++i) {
        counts[i] = 1;
        for (int j = i + 1; j < size; ++j) {
            if (array[i] == array[j]) {
                counts[i]++;
            }
        }
        if (counts[i] > max_count) {
            max_count = counts[i];
            mode = array[i];
        }
    }

    free(counts);
    return mode;
}

// Function to calculate the mean of an array (for regression)
double calculate_mean(const double *array, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum / size;
}

KNNModel KNNClassifier(const Matrix *X, const Matrix *y, int k) {
    if (X->rows != y->rows) {
        fprintf(stderr, "Error in KNN, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    if (k <= 0) {
        fprintf(stderr, "Error in KNN, k must be positive!");
        exit(EXIT_FAILURE);
    }

    KNNModel model;

    model.data.X = Matrix_clone(X);
    model.data.y = Matrix_clone(y);
    model.k = k;

    model.is_classification = true;

    return model;
}

KNNModel KNNRegressor(const Matrix *X, const Matrix *y, int k) {
    if (X->rows != y->rows) {
        fprintf(stderr, "Error in KNN, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    if (k <= 0) {
        fprintf(stderr, "Error in KNN, k must be positive!");
        exit(EXIT_FAILURE);
    }

    KNNModel model;

    model.data.X = Matrix_clone(X);
    model.data.y = Matrix_clone(y);
    model.k = k;

    model.is_classification = false;

    return model;
}

void KNN_free(KNNModel model) {
    Matrix_free(model.data.X);
    Matrix_free(model.data.y);
}

void KNN_append_data(KNNModel *model, const Matrix *X_new, const Matrix *y_new) {
    if (X_new->rows != y_new->rows) {
        fprintf(stderr, "Error: Mismatched rows in X_new and y_new.\n");
        exit(EXIT_FAILURE);
    }

    if (X_new->cols != model->data.X.cols) {
        fprintf(stderr, "Error: Column mismatch. X_new should have the same number of columns as the existing X.\n");
        exit(EXIT_FAILURE);
    }

    int new_num_samples = model->data.X.rows + X_new->rows;
    Matrix new_X = Matrix_zeros(new_num_samples, model->data.X.cols);
    Matrix new_y = Matrix_zeros(new_num_samples, 1);

    for (int i = 0; i < model->data.X.rows; ++i) {
        for (int j = 0; j < model->data.X.cols; ++j) {
            new_X.data[i][j] = model->data.X.data[i][j];
        }
        new_y.data[i][0] = model->data.y.data[i][0];
    }

    for (int i = 0; i < X_new->rows; ++i) {
        for (int j = 0; j < X_new->cols; ++j) {
            new_X.data[model->data.X.rows + i][j] = X_new->data[i][j];
        }
        new_y.data[model->data.X.rows + i][0] = y_new->data[i][0];
    }

    Matrix_free(model->data.X);
    Matrix_free(model->data.y);

    model->data.X = new_X;
    model->data.y = new_y;
}

Matrix KNN_predict(const KNNModel *model, const Matrix *x_new) {
    int num_new_samples = x_new->rows;
    int num_features = x_new->cols;
    int num_existing_samples = model->data.X.rows;
    int k = model->k;

    Matrix predictions = Matrix_zeros(num_new_samples, 1);

    for (int i = 0; i < num_new_samples; ++i) {
        double (*distances)[2] = malloc(num_existing_samples * sizeof(*distances));
        if (distances == NULL) {
            fprintf(stderr, "Memory allocation error in KNN_predict.\n");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < num_existing_samples; ++j) {
            distances[j][0] = euclidean_distance(x_new->data[i], model->data.X.data[j], num_features);
            distances[j][1] = j;  
        }

        qsort(distances, num_existing_samples, sizeof(*distances), compare_distances);

        double *neighbor_labels = malloc(k * sizeof(double));
        if (neighbor_labels == NULL) {
            fprintf(stderr, "Memory allocation error in KNN_predict.\n");
            free(distances);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < k; ++j) {
            int neighbor_index = (int)distances[j][1];
            neighbor_labels[j] = model->data.y.data[neighbor_index][0];
        }

        if (model->is_classification) {
            predictions.data[i][0] = find_mode(neighbor_labels, k);
        } else {
            predictions.data[i][0] = calculate_mean(neighbor_labels, k);
        }

        free(distances);
        free(neighbor_labels);
    }

    return predictions;
}

double sigmoid(double z) {
    return 1 / (1 + pow(EULER_NUMBER, -z));
}

LogisticRegressionModel LogisticRegression(const Matrix *X, const Matrix *y) {
    if (X->rows != y->rows || y->cols != 1) {
        fprintf(stderr, "Error in LinearRegression, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(X->rows, X->cols+1);
    for (int i = 0; i < X->rows; ++i) {
        XPadded.data[i][0] = 1;
    }

    for (int i = 0; i < X->rows; ++i) {
        for (int j = 0; j < X->cols; ++j) {
            XPadded.data[i][j+1] = X->data[i][j];
        }
    }

    LogisticRegressionModel model;

    model.data.X = XPadded;
    model.data.y = Matrix_clone(y);
    model.trained = false;

    model.params = Matrix_zeros(XPadded.cols, 1);

    return model;
}

void LogisticRegression_train(LogisticRegressionModel *model) {
    int m = model->data.X.rows;
    int n = model->data.X.cols;
    double **X = model->data.X.data;
    double **y = model->data.y.data;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double *predictions = (double *)malloc(m * sizeof(double));
        double *errors = (double *)malloc(m * sizeof(double));

        for (int i = 0; i < m; ++i) {
            double z = 0.0;
            for (int j = 0; j < n; ++j) {
                z += X[i][j] * model->params.data[j][0];
            }
            predictions[i] = sigmoid(z);
            errors[i] = predictions[i] - y[i][0];
        }

        double *grad = (double *)calloc(n, sizeof(double));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                grad[j] += errors[i] * X[i][j];
            }
        }

        for (int j = 0; j < n; ++j) {
            model->params.data[j][0] -= LEARNING_RATE * grad[j] / m;
        }

        free(predictions);
        free(errors);
        free(grad);
    }

    model->trained = true;
}

Matrix LogisticRegression_predict(const LogisticRegressionModel *model, const Matrix *X_new) {
    if (!model->trained) {
        fprintf(stderr, "Error in LogisticRegression_predict: model not trained!\n");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(X_new->rows, X_new->cols);
    for (int i = 0; i < X_new->rows; ++i) {
        XPadded.data[i][0] = 1.0;  
        for (int j = 0; j < X_new->cols+1; ++j) {
            XPadded.data[i][j + 1] = X_new->data[i][j];
        }
    }

    Matrix probabilities = Matrix_zeros(X_new->rows, 1);
    for (int i = 0; i < X_new->rows; ++i) {
        double z = 0.0;
        for (int j = 0; j < X_new->cols+1; ++j) {
            z += XPadded.data[i][j] * model->params.data[j][0];
        }
        probabilities.data[i][0] = sigmoid(z);
    }

    Matrix_free(XPadded);
    return probabilities;
}
