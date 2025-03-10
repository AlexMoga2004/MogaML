#include "../include/supervised.h"

/**
 * @brief Macro to output error message to stdout
 */
#define ERROR(fmt, ...)                                                        \
    do {                                                                       \
        fprintf(stderr, fmt, ##__VA_ARGS__);                                   \
    } while (0)

/**
 * @brief Calculate the euclidean distance between two vector arrays
 *
 * @param vec1 First vector array
 * @param vec2 Second vector array
 * @param length Length of arrays
 * @return double Euclidean distance
 */
static double euclidean_distance(const double *vec1, const double *vec2,
                                 int length) {
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/**
 * @brief Compare distances
 *
 * @param a First Value
 * @param b Second Value
 * @return  1 if a > b
 * @return -1 if a < b
 * @return  0 if a = b
 */
static int compare_distances(const void *a, const void *b) {
    double diff = ((double *)a)[0] - ((double *)b)[0];
    return (diff > 0) - (diff < 0);
}

/**
 * @brief Find the mode of a double array
 *
 * @param array
 * @param size
 * @return double Mode
 */
static double find_mode(const double *array, int size) {
    int *counts = calloc(size, sizeof(int));
    if (counts == NULL) {
        ERROR("Memory allocation error in find_mode.\n");
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

/**
 * @brief Calculate the mean of a double array
 *
 * @param array
 * @param size
 * @return double Mean
 */
static double calculate_mean(const double *array, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum / size;
}

/**
 * @brief Count the number of values per line in a CSV file (assuming same on
 * each line)
 *
 * @param filename
 * @return int count
 */
static int count_columns(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        ERROR("Error opening file: %s\n", filename);
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

/**
 * @brief Find the nearest centroid to a given point (embedded as a double
 * array)
 *
 * @param point Array of values representing point in euclidean space
 * @param centroids matrix containing centroid coordinates as rows
 * @return index of closes centroid
 */
int Supervised_find_nearest_centroid(const double *point,
                                     const Matrix *centroids) {
    int nearest = 0;
    double min_dist = DBL_MAX;
    for (int i = 0; i < centroids->rows; ++i) {
        double dist =
            euclidean_distance(point, centroids->data[i], centroids->cols);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    return nearest;
}

/**
 * @brief Load Comma Separated Values (CSV) into a LabelledData structure,
 * assuming the last element is the label
 *
 * @param filename
 * @return LabelledData parsed data
 * @throw Exception file must exist, and be structured appropriately
 */
LabelledData Supervised_read_csv(const char *filename) {
    LabelledData data;

    FILE *file = fopen(filename, "r");
    if (!file) {
        ERROR("Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

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

    data.X = Matrix_zeros(num_lines, num_features);
    data.y = Matrix_zeros(num_lines, 1);

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

/**
 * @brief Construct a Linear Regression model with no regularisation
 *
 * @param X input data in a Matrix
 * @param y output labels in a Vector (embedded as a Matrix)
 * @return LinearRegressionModel with default computation mode set to ALGEBRAIC
 * @throw Exception when X->rows != y->rows
 * @note Input Matrix is padded with a 1-column for the bias term
 */
LinearRegressionModel LinearRegression(const Matrix *X, const Matrix *y) {
    if (X->rows != y->rows || y->cols != 1) {
        ERROR("Error in LinearRegression, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(X->rows, X->cols + 1);
    for (int i = 0; i < X->rows; ++i) {
        XPadded.data[i][0] = 1;
    }

    for (int i = 0; i < X->rows; ++i) {
        for (int j = 0; j < X->cols; ++j) {
            XPadded.data[i][j + 1] = X->data[i][j];
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
LinearRegressionModel RidgeRegression(const Matrix *X, const Matrix *y,
                                      double lambda) {
    if (lambda <= 0) {
        ERROR("Error in RidgeRegression, λ ≤ 0!");
        exit(EXIT_FAILURE);
    }

    LinearRegressionModel model = LinearRegression(X, y);
    model.loss_function = RidgeRegression_default_loss();

    model.hyper_params = Matrix_zeros(1, 1);
    model.hyper_params.data[0][0] = lambda;

    return model;
}

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
LinearRegressionModel LassoRegression(const Matrix *X, const Matrix *y,
                                      double lambda) {
    if (lambda <= 0) {
        ERROR("Error in LassoRegression, λ ≤ 0!");
        exit(EXIT_FAILURE);
    }

    LinearRegressionModel model = LinearRegression(X, y);
    model.loss_function = LassoRegression_default_loss();

    model.hyper_params = Matrix_zeros(1, 1);
    model.hyper_params.data[0][0] = lambda;

    return model;
}

/**
 * @brief Train a linear regression model
 *
 * @param model Model to be trained
 */
void LinearRegression_train(LinearRegressionModel *model) {
    if (model->mode == ALGEBRAIC) {
        // May throw exception if loss_function has no closed form exact
        // solution
        model->params = model->loss_function.exact_optimum(
            &model->data.X, &model->data.y, &model->hyper_params);
    } else if (model->mode == BATCH) {
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            Matrix gradients =
                model->loss_function.grad(&model->data.X, &model->data.y,
                                          &model->params, &model->hyper_params);
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
        int num_batches = model->data.X.rows / MINIBATCH_SIZE;
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            for (int batch = 0; batch < num_batches; ++batch) {
                int start = batch * MINIBATCH_SIZE;
                int end = start + MINIBATCH_SIZE;

                Matrix X_batch = Matrix_slice_rows(&model->data.X, start, end);
                Matrix y_batch = Matrix_slice_rows(&model->data.y, start, end);

                LinearRegressionModel mini_model = *model;
                mini_model.data.X = X_batch;
                mini_model.data.y = y_batch;
                Matrix gradients = model->loss_function.grad(
                    &mini_model.data.X, &mini_model.data.y, &mini_model.params,
                    &mini_model.hyper_params);

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
        ERROR("Error in LinearRegression_train, undefined computation mode!\n");
        exit(EXIT_FAILURE);
    }

    model->trained = true;
}

/**
 * @brief Set the computation mode of a Linear Regression Model
 *
 * @param model The model to be changed
 * @param mode Desired computation mode
 * @note Some LossFunctions do not have an algebraic solution, and may throw
 * exceptions when trained with computation mode ALGEBRAIC
 */
void LinearRegression_set_mode(LinearRegressionModel *model,
                               enum ComputationMode mode) {
    model->mode = mode;
    model->trained = false;
}

/**
 * @brief Free allocated memory for a LinearRegressionModel
 *
 * @param model Model to be freed
 * @note Further calls to model may segfault
 */
void LinearRegression_free(LinearRegressionModel model) {
    Matrix_free(model.data.X);
    Matrix_free(model.data.y);
    Matrix_free(model.params);
}

/**
 * @brief Change the loss function
 *
 * @param model model to be changed
 * @param loss_function New loss function
 * @note model needs to be retrained before predictions are made
 */
void LinearRegression_set_loss(LinearRegressionModel *model,
                               LossFunction loss_function) {
    model->loss_function = loss_function;
    model->trained = false;
}

/**
 * @brief Mean Squared Error (MSE) LossFunction with no regularisation
 *
 * @return LossFunction
 */
LossFunction LinearRegression_default_loss() {
    LossFunction loss_function;
    loss_function.exact_optimum = LinearRegression_exact_optimum;
    loss_function.loss = LinearRegression_compute_mse;
    loss_function.grad = LinearRegression_compute_gradient;

    return loss_function;
}

/**
 * @brief Mean Squared Error (MSE) LossFunction with Ridge regularisation
 * (hyper-parameter lambda = model->hyper_params[0][0])
 *
 * @return LossFunction
 */
LossFunction RidgeRegression_default_loss() {
    LossFunction loss_function;
    loss_function.exact_optimum = RidgeRegression_exact_optimum;
    loss_function.loss = RidgeRegression_compute_mse;
    loss_function.grad = RidgeRegression_compute_gradient;

    return loss_function;
}

/**
 * @brief Mean Squared Error (MSE) LossFunction with Lasso regularisation
 * (hyper-parameter lambda = model->hyper_params[0][0])
 *
 * @return LossFunction
 * @note Cannot be trained with computation mode set to ALGEBRAIC
 */
LossFunction LassoRegression_default_loss() {
    LossFunction loss_function;
    loss_function.exact_optimum = LassoRegression_exact_optimum;
    loss_function.loss = LassoRegression_compute_mse;
    loss_function.grad = LassoRegression_compute_gradient;

    return loss_function;
}

/**
 * @brief Calculate the Linear Regression Estimate for new data
 *
 * @param model Linear Regression Model to predict with
 * @param x_new Matrix containing the new (unpadded) data as rows
 * @return Vector (embedded as Matrix) containing the predicted labels
 */
Matrix LinearRegression_predict(const LinearRegressionModel *model,
                                const Matrix *x_new) {
    if (!model->trained) {
        ERROR("Error in Supervised_predict, model not trained!");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(x_new->rows, x_new->cols + 1);
    for (int i = 0; i < x_new->rows; ++i) {
        XPadded.data[i][0] = 1;
    }

    for (int i = 0; i < x_new->rows; ++i) {
        for (int j = 0; j < x_new->cols; ++j) {
            XPadded.data[i][j + 1] = x_new->data[i][j];
        }
    }

    return Matrix_multiply(&XPadded, &model->params);
}

/**
 * @brief Compute Mean Squared Estimate (MSE)
 *
 * @param X Matrix with input data
 * @param y Vector (embedded in a Matrix) with correct labels
 * @param params Weights and bias used to make prediction
 * @param hyper_params Unneeded
 * @return double Mean Squared Error
 */
double LinearRegression_compute_mse(const Matrix *X, const Matrix *y,
                                    const Matrix *params,
                                    const Matrix *hyper_params) {
    Matrix predictions = Matrix_multiply(X, params);
    if (predictions.rows != y->rows || predictions.cols != y->cols) {
        ERROR("Error in Supervised_compute_mse, dimension mismatch!");
        exit(EXIT_FAILURE);
    }
    double mse = 0.0;

    for (int i = 0; i < predictions.rows; ++i) {
        mse += pow((predictions.data[i][0] - y->data[i][0]), 2);
    }

    Matrix_free(predictions);
    return mse / (double)y->rows;
}

/**
 * @brief Compute Mean Squared Estimate (MSE) with Ridge Regularisation
 *
 * @param X Matrix with input data
 * @param y Vector (embedded in a Matrix) with correct labels
 * @param params Weights and bias used to make prediction
 * @param hyper_params Contains lambda in data[0][0]
 * @return double Mean Squared Error
 */
double RidgeRegression_compute_mse(const Matrix *X, const Matrix *y,
                                   const Matrix *params,
                                   const Matrix *hyper_params) {
    if (hyper_params->rows != 1 && hyper_params->cols != 1) {
        ERROR("Error in RidgeRegression_compute_gradient, hyper_param must "
              "contain "
              "single value!");
        exit(EXIT_FAILURE);
    }

    if (hyper_params->data[0][0] <= 0) {
        ERROR("Error in RidgeRegression_compute_gradient, lambda must be "
              "positive");
        exit(EXIT_FAILURE);
    }

    double mse = LinearRegression_compute_mse(X, y, params, hyper_params);
    mse +=
        (hyper_params->data[0][0] * Vector_norm(params, 2)) / (double)y->rows;

    return mse;
}

/**
 * @brief Compute Mean Squared Estimate (MSE) with Lasso Regularisation
 *
 * @param X Matrix with input data
 * @param y Vector (embedded in a Matrix) with correct labels
 * @param params Weights and bias used to make prediction
 * @param hyper_params Contains lambda in data[0][0]
 * @return double Mean Squared Error
 */
double LassoRegression_compute_mse(const Matrix *X, const Matrix *y,
                                   const Matrix *params,
                                   const Matrix *hyper_params) {
    if (hyper_params->rows != 1 && hyper_params->cols != 1) {
        ERROR("Error in RidgeRegression_compute_gradient, hyper_param must "
              "contain "
              "single value!");
        exit(EXIT_FAILURE);
    }

    if (hyper_params->data[0][0] <= 0) {
        ERROR("Error in RidgeRegression_compute_gradient, lambda must be "
              "positive");
        exit(EXIT_FAILURE);
    }

    double mse = LinearRegression_compute_mse(X, y, params, hyper_params);
    mse +=
        (hyper_params->data[0][0] * Vector_norm(params, 1)) / (double)y->rows;

    return mse;
}

/**
 * @brief Computes the global optimum algebraically for Linear Regression with
 * no regularisation
 *
 * @param X Input features
 * @param y Input labels
 * @param hyper_params Unneeded
 * @return Matrix Global optimum value of params
 */
Matrix LinearRegression_exact_optimum(const Matrix *X, const Matrix *y,
                                      const Matrix *hyper_params) {
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

/**
 * @brief Computes the global optimum algebraically for Linear Regression with
 * Ridge regularisation
 *
 * @param X Input features
 * @param y Input labels
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix Global optimum value of params
 */
Matrix RidgeRegression_exact_optimum(const Matrix *X, const Matrix *y,
                                     const Matrix *hyper_params) {
    if (hyper_params->rows != 1 && hyper_params->cols != 1) {
        ERROR("Error in RidgeRegression_compute_gradient, hyper_param must "
              "contain "
              "single value!");
        exit(EXIT_FAILURE);
    }

    if (hyper_params->data[0][0] <= 0) {
        ERROR("Error in RidgeRegression_compute_gradient, lambda must be "
              "positive");
        exit(EXIT_FAILURE);
    }

    Matrix XT = Matrix_transpose(X);
    Matrix XTX = Matrix_multiply(&XT, X);

    Matrix I = Matrix_identity(XTX.rows);
    printf("%f\n", hyper_params->data[0][0]);
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

/**
 * @brief Placeholder function, throws exception since exact optimum can't be
 * computed for Lasso Regression
 *
 * @param X Input features
 * @param y Input labels
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix Global optimum value of params
 * @throw Exception
 */
Matrix LassoRegression_exact_optimum(const Matrix *X, const Matrix *y,
                                     const Matrix *hyper_params) {
    ERROR(
        "Error in LassoRegression_exact_optimum, closed form solution does not "
        "exist! (HINT: don't use ALGEBRAIC computation mode with Lasso)");
    exit(EXIT_FAILURE);
}

/**
 * @brief Computes the gradient of the MSE Loss Function
 *
 * @param X Input features
 * @param y Input labels
 * @param params Current parameters
 * @param hyper_params Unneeded
 * @return Matrix derivative of the loss function wrt the parameters
 */
Matrix LinearRegression_compute_gradient(const Matrix *X, const Matrix *y,
                                         const Matrix *params,
                                         const Matrix *hyper_params) {
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

/**
 * @brief Compute a sub-gradient of the MSE Loss Function with Ridge
 * regularisation
 *
 * @param X Input features
 * @param y Input labels
 * @param params Current parameters
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix derivative of the loss function wrt the parameters
 */
Matrix RidgeRegression_compute_gradient(const Matrix *X, const Matrix *y,
                                        const Matrix *params,
                                        const Matrix *hyper_params) {
    if (hyper_params->rows != 1 || hyper_params->cols != 1) {
        ERROR("Error in RidgeRegression_compute_gradient: hyper_params must "
              "contain a single value (lambda)!");
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

/**
 * @brief Computes a sub-gradient of the MSE Loss Function with Lasso
 * regularisation
 *
 * @param X Input features
 * @param y Input labels
 * @param params Current parameters
 * @param hyper_params Contains lambda in data[0][0]
 * @return Matrix derivative of the loss function wrt the parameters
 */
Matrix LassoRegression_compute_gradient(const Matrix *X, const Matrix *y,
                                        const Matrix *params,
                                        const Matrix *hyper_params) {
    if (hyper_params->rows != 1 || hyper_params->cols != 1) {
        ERROR("Error in LassoRegression_compute_gradient: hyper_params must "
              "contain a single value (lambda)!");
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
        double sign = (params->data[i][0] > 0)   ? 1.0
                      : (params->data[i][0] < 0) ? -1.0
                                                 : 0.0; // subgradient
        gradient.data[i][0] += lambda * sign / y->rows;
    }

    Matrix_free(predictions);
    Matrix_free(errors);
    Matrix_free(XT);

    return gradient;
}

/**
 * @brief Constructs a KNN model using the majority-vote prediction
 *
 * @param k Number of neighbours used by the model
 * @param X Input features
 * @param y Input labels
 * @return KNNModel
 */
KNNModel KNNClassifier(unsigned int k, const Matrix *X, const Matrix *y) {
    if (X->rows != y->rows) {
        ERROR("Error in KNN, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    if (k <= 0) {
        ERROR("Error in KNN, k must be positive!");
        exit(EXIT_FAILURE);
    }

    KNNModel model;

    model.data.X = Matrix_clone(X);
    model.data.y = Matrix_clone(y);
    model.k = k;

    model.is_classification = true;

    return model;
}

/**
 * @brief Constructs a KNN model using the majority-vote prediction
 *
 * @param k Number of neighbours used by the model
 * @param X Input features
 * @param y Input labels
 * @return KNNModel
 */
KNNModel KNNRegressor(unsigned int k, const Matrix *X, const Matrix *y) {
    if (X->rows != y->rows) {
        ERROR("Error in KNN, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    if (k <= 0) {
        ERROR("Error in KNN, k must be positive!");
        exit(EXIT_FAILURE);
    }

    KNNModel model;

    model.data.X = Matrix_clone(X);
    model.data.y = Matrix_clone(y);
    model.k = k;

    model.is_classification = false;

    return model;
}

/**
 * @brief Free memory allocated by KNN Model
 *
 * @param model
 */
void KNN_free(KNNModel model) {
    Matrix_free(model.data.X);
    Matrix_free(model.data.y);
}

/**
 * @brief Add data to existing KNN model
 *
 * @param model Existing model
 * @param X New input features
 * @param y New input labels
 */
void KNN_append_data(KNNModel *model, const Matrix *X_new,
                     const Matrix *y_new) {
    if (X_new->rows != y_new->rows) {
        ERROR("Error: Mismatched rows in X_new and y_new.\n");
        exit(EXIT_FAILURE);
    }

    if (X_new->cols != model->data.X.cols) {
        ERROR("Error: Column mismatch. X_new should have the same number of "
              "columns as the existing X.\n");
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

/**
 * @brief Predict the labels of unseen data using the KNN algorithm
 *
 * @param model
 * @param x_new
 * @return Matrix
 */
Matrix KNN_predict(const KNNModel *model, const Matrix *x_new) {
    int num_new_samples = x_new->rows;
    int num_features = x_new->cols;
    int num_existing_samples = model->data.X.rows;
    int k = model->k;

    Matrix predictions = Matrix_zeros(num_new_samples, 1);

    for (int i = 0; i < num_new_samples; ++i) {
        double(*distances)[2] =
            malloc(num_existing_samples * sizeof(*distances));
        if (distances == NULL) {
            ERROR("Memory allocation error in KNN_predict.\n");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < num_existing_samples; ++j) {
            distances[j][0] = euclidean_distance(
                x_new->data[i], model->data.X.data[j], num_features);
            distances[j][1] = j;
        }

        qsort(distances, num_existing_samples, sizeof(*distances),
              compare_distances);

        double *neighbor_labels = malloc(k * sizeof(double));
        if (neighbor_labels == NULL) {
            ERROR("Memory allocation error in KNN_predict.\n");
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

static double sigmoid(double z) { return 1 / (1 + pow(EULER_NUMBER, -z)); }

/**
 * @brief Construct a Logistic Regression Model
 *
 * @param X Input features
 * @param y Input labels
 * @return LogisticRegressionModel
 */
LogisticRegressionModel LogisticRegression(const Matrix *X, const Matrix *y) {
    if (X->rows != y->rows || y->cols != 1) {
        ERROR("Error in LinearRegression, dimension mismatch!");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(X->rows, X->cols + 1);
    for (int i = 0; i < X->rows; ++i) {
        XPadded.data[i][0] = 1;
    }

    for (int i = 0; i < X->rows; ++i) {
        for (int j = 0; j < X->cols; ++j) {
            XPadded.data[i][j + 1] = X->data[i][j];
        }
    }

    LogisticRegressionModel model;

    model.data.X = XPadded;
    model.data.y = Matrix_clone(y);
    model.trained = false;

    model.params = Matrix_zeros(XPadded.cols, 1);

    return model;
}

/**
 * @brief Train a Logistic Regression model
 *
 * @param model to be trained
 */
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

/**
 * @brief Predict the labels of unseen data
 *
 * @param model Model to be used
 * @param X_new Data to be predicted
 * @return Vector (embedded in a Matrix) containing the predicted labels
 * @throw Exception requires model to be trained
 */
Matrix LogisticRegression_predict(const LogisticRegressionModel *model,
                                  const Matrix *X_new) {
    if (!model->trained) {
        ERROR("Error in LogisticRegression_predict: model not trained!\n");
        exit(EXIT_FAILURE);
    }

    Matrix XPadded = Matrix_zeros(X_new->rows, X_new->cols + 1);
    for (int i = 0; i < X_new->rows; ++i) {
        XPadded.data[i][0] = 1.0; // Bias
        for (int j = 0; j < X_new->cols; ++j) {
            XPadded.data[i][j + 1] = X_new->data[i][j];
        }
    }

    Matrix probabilities = Matrix_zeros(X_new->rows, 1);

    for (int i = 0; i < X_new->rows; ++i) {
        double z = 0.0;
        for (int j = 0; j < XPadded.cols; ++j) {
            z += XPadded.data[i][j] * model->params.data[j][0];
        }
        probabilities.data[i][0] = sigmoid(z);
    }

    Matrix_free(XPadded);

    return probabilities;
}

/**
 * @brief Create a Naive Bayes Classifier (NBC) model
 *
 * @param X Input features
 * @param y Input labels
 * @return GaussianNBCModel
 */
GaussianNBCModel GaussianNBC(const Matrix *X, const Matrix *y) {
    int num_samples = X->rows;
    int num_features = X->cols;

    int num_classes = 0;
    for (int i = 0; i < num_samples; ++i) {
        if ((int)y->data[i][0] + 1 > num_classes) {
            num_classes = (int)y->data[i][0] + 1;
        }
    }

    GaussianNBCModel model;
    model.num_classes = num_classes;
    model.means = Matrix_zeros(num_classes, num_features);
    model.variances = Matrix_zeros(num_classes, num_features);
    model.priors = Matrix_zeros(num_classes, 1);

    int *class_counts = (int *)calloc(num_classes, sizeof(int));
    for (int i = 0; i < num_samples; ++i) {
        int class_label = (int)y->data[i][0];
        class_counts[class_label]++;
    }

    for (int c = 0; c < num_classes; ++c) {
        model.priors.data[c][0] = (double)class_counts[c] / num_samples;
    }

    for (int i = 0; i < num_samples; ++i) {
        int class_label = (int)y->data[i][0];
        for (int j = 0; j < num_features; ++j) {
            model.means.data[class_label][j] += X->data[i][j];
        }
    }
    for (int c = 0; c < num_classes; ++c) {
        for (int j = 0; j < num_features; ++j) {
            model.means.data[c][j] /= class_counts[c];
        }
    }

    for (int i = 0; i < num_samples; ++i) {
        int class_label = (int)y->data[i][0];
        for (int j = 0; j < num_features; ++j) {
            double diff = X->data[i][j] - model.means.data[class_label][j];
            model.variances.data[class_label][j] += diff * diff;
        }
    }
    for (int c = 0; c < num_classes; ++c) {
        for (int j = 0; j < num_features; ++j) {
            model.variances.data[c][j] /= class_counts[c];
        }
    }

    free(class_counts);
    model.trained = true;
    return model;
}

/**
 * @brief Use the classifier to predict the labels of unseen data
 *
 * @param model Model to be used
 * @param X_new Unseen data
 * @return Matrix Predicted labels in Vector form
 */
Matrix GaussianNBC_predict(const GaussianNBCModel *model, const Matrix *X_new) {
    if (!model->trained) {
        ERROR("Error in GaussianNaiveBayes_predict: model not trained!\n");
        exit(EXIT_FAILURE);
    }

    int num_samples = X_new->rows;
    int num_features = X_new->cols;
    int num_classes = model->num_classes;

    Matrix predictions = Matrix_zeros(num_samples, 1);

    for (int i = 0; i < num_samples; ++i) {
        double max_posterior = -INFINITY;
        int best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            double log_prior = log(model->priors.data[c][0]);
            double log_likelihood = 0.0;
            for (int j = 0; j < num_features; ++j) {
                double mean = model->means.data[c][j];
                double variance = model->variances.data[c][j];
                double x = X_new->data[i][j];
                log_likelihood += -0.5 * log(2 * M_PI * variance) -
                                  (x - mean) * (x - mean) / (2 * variance);
            }
            double log_posterior = log_prior + log_likelihood;
            if (log_posterior > max_posterior) {
                max_posterior = log_posterior;
                best_class = c;
            }
        }
        predictions.data[i][0] = best_class;
    }

    return predictions;
}
