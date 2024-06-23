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

    return model;
}

void Supervised_train(LinearRegressionModel *model) {
    if (model->mode == ALGEBRAIC) {
        Matrix XT = Matrix_transpose(&model->data.X);
        Matrix XTX = Matrix_multiply(&XT, &model->data.X);
        Matrix XTXI = Matrix_inverse(&XTX);
        Matrix hat = Matrix_multiply(&XTXI, &XT);
        Matrix result = Matrix_multiply(&hat, &model->data.y);

        Matrix_free(XT);
        Matrix_free(XTX);
        Matrix_free(XTXI);
        Matrix_free(hat);
        model->params = result;
    } else if (model->mode == BATCH) {
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            Matrix gradients = Supervised_compute_gradient(model);
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
                Matrix gradients = Supervised_compute_gradient(&mini_model);

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
        fprintf(stderr, "Error in Supervised_train, undefined computation mode!\n");
        exit(EXIT_FAILURE);
    }

    model->trained = true;
}

void Supervised_set_mode(LinearRegressionModel *model, enum ComputationMode mode){
    model->mode = mode;
}

void Supervised_free(LinearRegressionModel model) {
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

double Supervised_compute_mse(const LinearRegressionModel *model) {
    Matrix predictions = Matrix_multiply(&model->data.X, &model->params);
    if (predictions.rows != model->data.y.rows || predictions.cols != model->data.y.cols) {
        fprintf(stderr, "Error in Supervised_compute_mse, dimension mismatch!");
        exit(EXIT_FAILURE);
    }
    double mse = 0.0;

    for (int i = 0; i < predictions.rows; ++i) {
        mse += pow((predictions.data[i][0] - model->data.y.data[i][0]), 2);
    }

    Matrix_free(predictions);
    return mse / (double) model->data.y.rows;
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

Matrix Supervised_compute_gradient(const LinearRegressionModel *model) {
    Matrix predictions = Matrix_multiply(&model->data.X, &model->params);
    Matrix errors = Matrix_sub(&predictions, &model->data.y);
    Matrix XT = Matrix_transpose(&model->data.X);
    Matrix gradient = Matrix_multiply(&XT, &errors);
    gradient = Matrix_scale(1.0 / model->data.y.rows, &gradient);

    Matrix_free(predictions);
    Matrix_free(errors);
    Matrix_free(XT);

    return gradient;
}
