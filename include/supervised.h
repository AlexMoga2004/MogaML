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
LinearRegressionModel RidgeRegression(const Matrix *X, const Matrix *y, double lambda);
LinearRegressionModel LassoRegression(const Matrix *X, const Matrix *y, double lambda);

void Supervised_train(LinearRegressionModel *model);
void Supervised_set_mode(LinearRegressionModel *model, enum ComputationMode mode);
void Supervised_free(LinearRegressionModel model);
void Supervised_set_loss(LinearRegressionModel *model, LossFunction loss_function);

LossFunction LinearRegression_default_loss();
LossFunction RidgeRegression_default_loss();
LossFunction LassoRegression_default_loss();

double LinearRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
double RidgeRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
double LassoRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

LabelledData Supervised_read_csv(const char* filename);

Matrix Supervised_predict(const LinearRegressionModel *model, const Matrix *x_new);

Matrix LinearRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_parameters);
Matrix LinearRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
Matrix RidgeRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_parameters);
Matrix RidgeRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
Matrix LassoRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);
Matrix LassoRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
