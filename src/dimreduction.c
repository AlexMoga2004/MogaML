#include "dimreduction.h"

/**
 * @brief Computes a low-rank approximation of a matrix using Principle Component Analysis
 * 
 * @param X Pointer to the Matrix
 * @param target_rank rank of the output Matrix
 * @return Matrix of rank target_rank
 */
Matrix PCA_Reduce(const Matrix *X, unsigned int target_rank) {
    Matrix result;

    // Compute SVD of X
    SVDResult svd = Matrix_svd(X);

    // Reduce Sigma matrix to target_rank by setting singular values beyond target_rank to zero
    for (int i = target_rank; i < svd.Sigma.rows; ++i) {
        svd.Sigma.data[i][i] = 0;
    }

    // Reconstruct the reduced matrix
    Matrix U_reduced = Matrix_submatrix(&svd.U, 0, svd.U.rows, 0, target_rank);
    Matrix Sigma_reduced = Matrix_submatrix(&svd.Sigma, 0, target_rank, 0, target_rank);
    Matrix VT_reduced = Matrix_submatrix(&svd.V, 0, target_rank, 0, svd.V.cols);

    // Compute the reduced matrix: result = U_reduced * Sigma_reduced * VT_reduced
    result = Matrix_multiply(&U_reduced, &Sigma_reduced);
    result = Matrix_multiply(&result, &VT_reduced);

    // Free allocated memory
    Matrix_free(U_reduced);
    Matrix_free(Sigma_reduced);
    Matrix_free(VT_reduced);
    Matrix_free(svd.U);
    Matrix_free(svd.Sigma);
    Matrix_free(svd.V);

    return result;
}

static double compute_chi_squared(const Matrix *X, const Matrix *y, int feature_index) {
    int num_samples = X->rows;
    int num_classes = 2; // For binary classification (0 and 1)

    double *observed = (double *)calloc(num_classes, sizeof(double));
    double *expected = (double *)calloc(num_classes, sizeof(double));

    if (!observed || !expected) {
        fprintf(stderr, "Memory allocation error in compute_chi_squared.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_samples; ++i) {
        int class_label = (int)y->data[i][0];
        observed[class_label]++;
    }

    double *bin_totals = (double *)calloc(2, sizeof(double));
    if (!bin_totals) {
        fprintf(stderr, "Memory allocation error in compute_chi_squared.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_samples; ++i) {
        if (X->data[i][feature_index] > 0.5) {
            bin_totals[1]++;
        } else {
            bin_totals[0]++;
        }
    }

    for (int i = 0; i < num_classes; ++i) {
        expected[i] = (observed[0] + observed[1]) * (bin_totals[i] / num_samples);
    }

    double chi_squared = 0.0;
    for (int i = 0; i < num_classes; ++i) {
        if (expected[i] != 0) {
            chi_squared += pow(observed[i] - expected[i], 2) / expected[i];
        }
    }

    free(observed);
    free(expected);
    free(bin_totals);

    return chi_squared;
}

static void sort_indices_by_scores(double *scores, int *indices, int num_features) {
    for (int i = 0; i < num_features - 1; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < num_features; ++j) {
            if (scores[j] > scores[max_idx]) {
                max_idx = j;
            }
        }

        double temp_score = scores[max_idx];
        scores[max_idx] = scores[i];
        scores[i] = temp_score;

        int temp_idx = indices[max_idx];
        indices[max_idx] = indices[i];
        indices[i] = temp_idx;
    }
}

/**
 * @brief Selects the most relevant input features using the chi-squared statistical test
 * 
 * @param X Pointer to original input data
 * @param y Pointer to output labels
 * @param target_features Number of features to select
 * @return Matrix with target_features input features
 */
Matrix ChiSquared_Reduce(const Matrix *X, const Matrix *y, unsigned int target_features) {
    int num_features = X->cols;

    if (target_features > num_features) {
        fprintf(stderr, "Error in ChiSquared_Reduce: target_features exceeds the number of original features.\n");
        exit(EXIT_FAILURE);
    }

    double *chi_squared_scores = (double *)calloc(num_features, sizeof(double));
    if (!chi_squared_scores) {
        fprintf(stderr, "Memory allocation error in ChiSquared_Reduce.\n");
        exit(EXIT_FAILURE);
    }

    for (int feature_index = 0; feature_index < num_features; ++feature_index) {
        double chi_squared_value = 0.0;

        chi_squared_value = compute_chi_squared(X, y, feature_index);
        chi_squared_scores[feature_index] = chi_squared_value;
    }

    int *selected_indices = (int *)malloc(num_features * sizeof(int));
    for (int i = 0; i < num_features; ++i) {
        selected_indices[i] = i;
    }
    sort_indices_by_scores(chi_squared_scores, selected_indices, num_features);

    Matrix result = Matrix_zeros(X->rows, target_features);
    for (int i = 0; i < target_features; ++i) {
        int selected_index = selected_indices[i];
        for (int j = 0; j < X->rows; ++j) {
            result.data[j][i] = X->data[j][selected_index];
        }
    }

    free(chi_squared_scores);
    free(selected_indices);

    return result;
} 