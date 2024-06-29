#include "dimreduction.h"

Matrix PCA_Reduce(const Matrix *X, unsigned int target_rank) {
    Matrix result;
    SVDResult svd = Matrix_svd(X);

    for (int i = target_rank; i < svd.Sigma.rows; ++i) {
        svd.Sigma.data[i][i] = 0;
    }

    Matrix VT = Matrix_transpose(&svd.V);

    result = Matrix_multiply(&svd.U, &svd.Sigma);
    result = Matrix_multiply(&result, &VT);

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