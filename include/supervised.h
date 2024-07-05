#pragma once
#include "matrix.h"
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define MINIBATCH_SIZE 32
#define TOLERANCE 1e-6

/**
 * @brief Represents the different ways a model can minimise a loss function
 */
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

/**
 * @brief Find the nearest centroid to a given point (embedded as a double array)
 * 
 * @param point Array of values representing point in euclidean space
 * @param centroids matrix containing centroid coordinates as rows
 * @return index of closes centroid
 */
int Supervised_find_nearest_centroid(const double *point, const Matrix *centroids);


/**
 * @brief Load Comma Separated Values (CSV) into a LabelledData structure, assuming the last element is the label
 * 
 * @param filename
 * @return LabelledData parsed data
 * @throw Exception file must exist, and be structured appropriately
 */
LabelledData Supervised_read_csv(const char* filename);

/*                      LINEAR REGRESSION w/ arbitrary loss                  */

/**
 * @brief Structure representing a loss function, with the algebraic optimal and the derivative
 */
typedef struct {
    Matrix (*exact_optimum) (const Matrix *X, const Matrix *y, const Matrix *hyper_params);              ///< Function to compute the global optimum algebraically (if applicable)
    double (*loss) (const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params); ///< Function to calculate the loss wrt the parameters
    Matrix (*grad) (const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params); ///< Derivative of the loss function wrt the parameters, used for numerical optimisation
} LossFunction;

/**
 * @brief Structure representing a Linear Regression model
 */
typedef struct {
    bool trained;

    LabelledData data;
    Matrix params;
    Matrix hyper_params;
    enum ComputationMode mode;

    LossFunction loss_function;
} LinearRegressionModel;

/**
 * @brief Construct a Linear Regression model with no regularisation
 * 
 * @param X input data in a Matrix
 * @param y output labels in a Vector (embedded as a Matrix)
 * @return LinearRegressionModel with default computation mode set to ALGEBRAIC
 * @throw Exception when X->rows != y->rows
 * @note Input Matrix is padded with a 1-column for the bias term
 */
LinearRegressionModel LinearRegression(const Matrix *X, const Matrix *y);

/**
 * @brief Construct a Linear Regression model with Ridge Regularisation
 * 
 * @param X input data in a Matrix
 * @param y output labels in a Vector (embedded as a Matrix)
 * @param lambda Hyper-parameter for Ridge loss
 * @return LinearRegressionModel with default computation mode set to ALGEBRAIC
 * @throw Exception when X->rows != y->rows
 * @note Input Matrix is padded with a 1-column for the bias term
 */
LinearRegressionModel RidgeRegression(const Matrix *X, const Matrix *y, double lambda);

/**
 * @brief Construct a Linear Regression model with Lasso Regularisation
 * 
 * @param X input data in a Matrix
 * @param y output labels in a Vector (embedded as a Matrix)
 * @param lambda Hyper-parameter for Ridge loss
 * @return LinearRegressionModel with default computation mode set to BATCH 
 * @throw Exception when X->rows != y->rows
 * @note No algebraic solution exists for lasso regression
 * @note Input Matrix is padded with a 1-column for the bias term
 */
LinearRegressionModel LassoRegression(const Matrix *X, const Matrix *y, double lambda);

/**
 * @brief Train a linear regression model
 * 
 * @param model LinearRegressionModel
 */
void LinearRegression_train(LinearRegressionModel *model);

/**
 * @brief Set the computation mode of a Linear Regression Model
 * 
 * @param model The model to be changed
 * @param mode Desired computation mode
 * @note Some LossFunctions do not have an algebraic solution, and may throw exceptions when trained with computation mode ALGEBRAIC
 */
void LinearRegression_set_mode(LinearRegressionModel *model, enum ComputationMode mode);

/**
 * @brief Free allocated memory for a LinearRegressionModel
 * 
 * @param model Model to be freed
 * @note Further calls to model may segfault
 */
void LinearRegression_free(LinearRegressionModel model);

/**
 * @brief Change the loss function
 * 
 * @param model model to be changed
 * @param loss_function New loss function
 * @note model needs to be retrained before predictions are made
 */
void LinearRegression_set_loss(LinearRegressionModel *model, LossFunction loss_function);

/**
 * @brief Mean Squared Error (MSE) LossFunction with no regularisation
 * 
 * @return LossFunction 
 */
LossFunction LinearRegression_default_loss();

/**
 * @brief Mean Squared Error (MSE) LossFunction with Ridge regularisation (hyper-parameter lambda = model->hyper_params[0][0])
 * 
 * @return LossFunction 
 */
LossFunction RidgeRegression_default_loss();

/**
 * @brief Mean Squared Error (MSE) LossFunction with Lasso regularisation (hyper-parameter lambda = model->hyper_params[0][0])
 * 
 * @return LossFunction 
 * @note Cannot be trained with computation mode set to ALGEBRAIC
 */
LossFunction LassoRegression_default_loss();

/**
 * @brief Calculate the Linear Regression Estimate for new data 
 * 
 * @param model Linear Regression Model to predict with
 * @param x_new Matrix containing the new (unpadded) data as rows
 * @return Vector (embedded as Matrix) containing the predicted labels
 */
Matrix LinearRegression_predict(const LinearRegressionModel *model, const Matrix *x_new);

/**
 * @brief Compute Mean Squared Estimate (MSE) 
 * 
 * @param X Matrix with input data
 * @param y Vector (embedded in a Matrix) with correct labels
 * @param params Weights and bias used to make prediction
 * @param hyper_params Unneeded
 * @return double Mean Squared Error
 */
double LinearRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/**
 * @brief Compute Mean Squared Estimate (MSE) with Ridge Regularisation
 * 
 * @param X Matrix with input data
 * @param y Vector (embedded in a Matrix) with correct labels
 * @param params Weights and bias used to make prediction
 * @param hyper_params Contains lambda in data[0][0]
 * @return double Mean Squared Error
 */
double RidgeRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/**
 * @brief Compute Mean Squared Estimate (MSE) with Lasso Regularisation
 * 
 * @param X Matrix with input data
 * @param y Vector (embedded in a Matrix) with correct labels
 * @param params Weights and bias used to make prediction
 * @param hyper_params Contains lambda in data[0][0]
 * @return double Mean Squared Error
 */
double LassoRegression_compute_mse(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/**
 * @brief Computes the global optimum algebraically for Linear Regression with no regularisation
 * 
 * @param X Input features
 * @param y Input labels
 * @param hyper_params Unneeded
 * @return Matrix Global optimum value of params
 */
Matrix LinearRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);

/**
 * @brief Computes the global optimum algebraically for Linear Regression with Ridge regularisation
 * 
 * @param X Input features
 * @param y Input labels
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix Global optimum value of params
 */
Matrix RidgeRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);

/**
 * @brief Placeholder function, throws exception since exact optimum can't be computed for Lasso Regression
 * 
 * @param X Input features
 * @param y Input labels
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix Global optimum value of params
 * @throw Exception
 */
Matrix LassoRegression_exact_optimum(const Matrix *X, const Matrix *y, const Matrix *hyper_params);

/**
 * @brief Computes the gradient of the MSE Loss Function
 * 
 * @param X Input features
 * @param y Input labels
 * @param params Current parameters
 * @param hyper_params Unneeded
 * @return Matrix derivative of the loss function wrt the parameters
 */
Matrix LinearRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/**
 * @brief Compute a sub-gradient of the MSE Loss Function with Ridge regularisation
 * 
 * @param X Input features
 * @param y Input labels
 * @param params Current parameters
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix derivative of the loss function wrt the parameters
 */
Matrix RidgeRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/**
 * @brief Computes a sub-gradient of the MSE Loss Function with Lasso regularisation
 * 
 * @param X Input features
 * @param y Input labels
 * @param params Current parameters
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix derivative of the loss function wrt the parameters
 */
Matrix LassoRegression_compute_gradient(const Matrix *X, const Matrix *y, const Matrix *params, const Matrix *hyper_params);

/*                  KNN                 */

/**
 * @brief Model for the K-Nearest-Neighbours algorithm
 */
typedef struct {
    int k;                          ///< Maximum number of neighbours to look at
    bool is_classification;         ///< Whether the model will take the majority, or the average over the neighbours

    LabelledData data;              ///< Input data
} KNNModel;


/**
 * @brief Constructs a KNN model using the majority-vote prediction
 * 
 * @param k Number of neighbours used by the model
 * @param X Input features
 * @param y Input labels
 * @return KNNModel 
 */
KNNModel KNNClassifier(unsigned int k, const Matrix *X, const Matrix *y);

/**
 * @brief Constructs a KNN model using the majority-vote prediction
 * 
 * @param k Number of neighbours used by the model
 * @param X Input features
 * @param y Input labels
 * @return KNNModel 
 */
KNNModel KNNRegressor(unsigned int k, const Matrix *X, const Matrix *y);

/**
 * @brief Free memory allocated by KNN Model
 * 
 * @param model 
 */
void KNN_free(KNNModel model);

/**
 * @brief Add data to existing KNN model
 * 
 * @param model Existing model
 * @param X New input features
 * @param y New input labels
 */
void KNN_append_data(KNNModel *model, const Matrix *X, const Matrix *y);

/**
 * @brief Predict the labels of unseen data using the KNN algorithm
 * 
 * @param model 
 * @param x_new 
 * @return Matrix 
 */
Matrix KNN_predict(const KNNModel *model, const Matrix *x_new);

/*                  LOGISTIC REGRESSION                 */
#define EULER_NUMBER 2.71828182845904523536

/**
 * @brief Structure to hold data used by Logistic Regression
 */
typedef struct {
    bool trained;

    LabelledData data;
    Matrix params;
} LogisticRegressionModel;

/**
 * @brief Construct a Logistic Regression Model
 * 
 * @param X Input features
 * @param y Input labels
 * @return LogisticRegressionModel 
 */
LogisticRegressionModel LogisticRegression(const Matrix *X, const Matrix *y);

/**
 * @brief Train a Logistic Regression model
 * 
 * @param model to be trained
 */
void LogisticRegression_train(LogisticRegressionModel *model);

/**
 * @brief Predict the labels of unseen data
 * 
 * @param model Model to be used
 * @param X_new Data to be predicted
 * @return Vector (embedded in a Matrix) containing the predicted labels
 * @throw Exception requires model to be trained
 */
Matrix LogisticRegression_predict(const LogisticRegressionModel *model, const Matrix *X_new);


/*                  Naive Bayes                     */

/**
 * @brief Structure to hold all data needed by a Gaussian Naive Bayes Classifier
 */
typedef struct {
    int num_classes;
    bool trained;

    Matrix means; 
    Matrix variances; 
    Matrix priors;
} GaussianNBCModel;

/**
 * @brief Create a Naive Bayes Classifier (NBC) model
 * 
 * @param X Input features
 * @param y Input labels
 * @return GaussianNBCModel 
 */
GaussianNBCModel GaussianNBC(const Matrix *X, const Matrix *y);

/**
 * @brief Use the classifier to predict the labels of unseen data
 * 
 * @param model Model to be used
 * @param X_new Unseen data
 * @return Matrix Predicted labels in Vector form
 */
Matrix GaussianNBC_predict(const GaussianNBCModel *model, const Matrix *X_new);
