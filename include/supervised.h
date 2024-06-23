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

// Arbitrary loss function w/ derivative
typedef struct {
    double (*loss) (const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyperParams);
    Matrix (*grad) (const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyperParams);
} LossFunction;

typedef struct {
    bool trained;

    LabelledData data;
    Matrix params;
    Matrix hyperParams;
    enum ComputationMode mode;

    LossFunction lossFunction;
} LinearRegressionModel;

LinearRegressionModel LinearRegression(const Matrix *X, const Matrix *y);

void Supervised_train(LinearRegressionModel *model);
void Supervised_set_mode(LinearRegressionModel *model, enum ComputationMode mode);
void Supervised_free(LinearRegressionModel model);

double Supervised_compute_mse(const LinearRegressionModel *model);

LabelledData Supervised_read_csv(const char* filename);

Matrix Supervised_predict(const LinearRegressionModel *model, const Matrix *x_new);
Matrix Supervised_compute_gradient(const LinearRegressionModel *model);

