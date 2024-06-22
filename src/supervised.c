#include "supervised.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void Supervised_train(LinearRegressionModel *model) {
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
    model->trained = true;
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

    return model;
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
