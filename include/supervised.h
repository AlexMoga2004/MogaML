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

/*                      LINEAR REGRESSION w/ arbitrary loss                  */

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

void LinearRegression_train(LinearRegressionModel *model);
void LinearRegression_set_mode(LinearRegressionModel *model, enum ComputationMode mode);
void LinearRegression_free(LinearRegressionModel model);
void LinearRegression_set_loss(LinearRegressionModel *model, LossFunction loss_function);

LossFunction LinearRegression_default_loss();
LossFunction RidgeRegression_default_loss();
LossFunction LassoRegression_default_loss();

LabelledData Supervised_read_csv(const char* filename);

Matrix Supervised_predict(const LinearRegressionModel *model, const Matrix *x_new);

// Default functions provided
double LinearRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
double RidgeRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
double LassoRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

Matrix LinearRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);
Matrix RidgeRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);
Matrix LassoRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);

Matrix LinearRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
Matrix RidgeRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);
Matrix LassoRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/*              KNN             */
typedef struct {
    int k;
    bool is_classification;

    LabelledData data;
} KNNModel;

KNNModel KNNClassifier(const Matrix *X, const Matrix *y, int k);
KNNModel KNNRegressor(const Matrix *X, const Matrix *y, int k);

void KNN_free(KNNModel model);
void KNN_append_data(KNNModel *model, const Matrix *X, const Matrix *y);

Matrix KNN_predict(const KNNModel *model, const Matrix *x_new);
