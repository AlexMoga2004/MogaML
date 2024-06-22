#pragma once
#include "matrix.h"
#include <stdbool.h>

enum ComputationMode {
    ALGABRAIC,
    BATCH,
    STOCHASTIC
};

typedef struct {
    Matrix X;
    Matrix y;
} LabelledData;

typedef struct {
    bool trained;
    LabelledData data;
    Matrix params;
} LinearRegressionModel;


void Supervised_train(LinearRegressionModel *model);
void Supervised_free(LinearRegressionModel model);

LabelledData Supervised_read_csv(const char* filename);

LinearRegressionModel LinearRegression(const Matrix *X, const Matrix *y);

Matrix Supervised_predict(const LinearRegressionModel *model, const Matrix *x_new);

