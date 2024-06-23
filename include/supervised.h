#pragma once
#include "matrix.h"
#include <stdbool.h>

#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define MINIBATCH_SIZE 32
#define TOLERANCE 1e-6

enum ComputationMode {
    ALGEBRAIC,
    BATCH,
    MINIBATCH,
    STOCHASTIC
};

typedef struct {
    Matrix X;
    Matrix y;
} LabelledData;

typedef struct {
    Matrix (*exact_optimum) (const Matrix *X, const Matrix *y, const Matrix *hyper_params); // May throw exception when no exact solution exists
    double (*loss) (const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
    Matrix (*grad) (const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
} LossFunction;

typedef struct {
    bool trained;

    LabelledData data;
    Matrix params;
    Matrix hyper_params;
    enum ComputationMode mode;

    LossFunction loss_function;
} LinearRegressionModel;

LinearRegressionModel LinearRegression(const Matrix *X, const Matrix *y);

void Supervised_train(LinearRegressionModel *model);
void Supervised_set_mode(LinearRegressionModel *model, enum ComputationMode mode);
void Supervised_free(LinearRegressionModel model);

double LinearRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

LabelledData Supervised_read_csv(const char* filename);

Matrix LinearRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_parameters);
Matrix Supervised_predict(const LinearRegressionModel *model, const Matrix *x_new);
Matrix LinearRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
