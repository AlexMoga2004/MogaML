#include "unsupervised.h"

#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define MINIBATCH_SIZE 32
#define TOLERANCE 1e-6

// Helper functions
static double euclidean_distance(const double *a, const double *b, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

static int find_nearest_centroid(const double *point, const Matrix *centroids) {
    int nearest = 0;
    double min_dist = DBL_MAX;
    for (int i = 0; i < centroids->rows; ++i) {
        double dist = euclidean_distance(point, centroids->data[i], centroids->cols);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    return nearest;
}

static int count_columns(const char *filename) {
    printf("Counting columns\n");
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
    printf("Done counting columns\n");
}

Matrix Unsupervised_read_csv(const char *filename) {
    printf("Is this shit even being executed????\n");

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
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

    int num_features = count_columns(filename);

    Matrix X = Matrix_zeros(num_lines, num_features);

    int i = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        char *token = strtok(buffer, ",");
        int j = 0;
        while (token != NULL && j < num_features) {
            X.data[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(file);
    return X;
}

KMeansModel KMeans(unsigned int k, const Matrix *X) {
    KMeansModel model;
    model.data = Matrix_clone(X);
    model.k = k;

    int n_samples = X->rows;
    int n_features = X->cols;
    
    // Initialise cluster_means to be k random unique items from X
    model.cluster_means = Matrix_zeros(k, n_features);
    model.labels = Matrix_zeros(n_samples, 1);

    int *chosen_indices = (int *)calloc(k, sizeof(int));
    for (int i = 0; i < k; ++i) chosen_indices[i] = -1;

    srand(time(NULL)); 

    for (int i = 0; i < k; ++i) {
        int index;
        bool unique;

        do {
            unique = true;
            index = rand() % n_samples;
            for (int j = 0; j < i; ++j) {
                if (chosen_indices[j] == index) {
                    unique = false;
                    break;
                }
            }
        } while (!unique);

        chosen_indices[i] = index;
        for (int j = 0; j < n_features; ++j) {
            model.cluster_means.data[i][j] = X->data[index][j];
        }
    }

    for (int i = 0; i < n_samples; ++i) {
        model.labels.data[i][0] = find_nearest_centroid(X->data[i], &model.cluster_means);
    }

    free(chosen_indices);
    return model;
}

void KMeans_train(KMeansModel *model) {
    int num_samples = model->data.rows;
    int num_features = model->data.cols;
    int k = model->k;

    Matrix new_cluster_means = Matrix_zeros(k, num_features);
    int *cluster_sizes = (int *)malloc(k * sizeof(int));
    int *labels = (int *)malloc(num_samples * sizeof(int));

    for (int iter = 0; iter < EPOCHS; ++iter) {
        Matrix_reset(&new_cluster_means);
        for (int i = 0; i < k; ++i) {
            cluster_sizes[i] = 0;
        }

        for (int i = 0; i < num_samples; ++i) {
            double min_distance = DBL_MAX;
            int min_index = -1;
            for (int j = 0; j < k; ++j) {
                double distance = euclidean_distance(model->data.data[i], model->cluster_means.data[j], num_features);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            labels[i] = min_index;

            // Update new cluster means
            for (int j = 0; j < num_features; ++j) {
                new_cluster_means.data[min_index][j] += model->data.data[i][j];
            }
            cluster_sizes[min_index]++;
        }

        for (int i = 0; i < k; ++i) {
            if (cluster_sizes[i] > 0) {
                for (int j = 0; j < num_features; ++j) {
                    new_cluster_means.data[i][j] /= cluster_sizes[i];
                }
            }
        }

        // Check for convergence
        double max_shift = 0.0;
        for (int i = 0; i < k; ++i) {
            double shift = euclidean_distance(new_cluster_means.data[i], model->cluster_means.data[i], num_features);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }
        if (max_shift < TOLERANCE) {
            break;
        }

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < num_features; ++j) {
                model->cluster_means.data[i][j] = new_cluster_means.data[i][j];
            }
        }
    }

    for (int i = 0; i < num_samples; ++i) {
        model->labels.data[i][0] = (double)labels[i];
    }

    Matrix_free(new_cluster_means);
    free(cluster_sizes);
    free(labels);

    model->trained = true;
}

Matrix KMeans_predict(const KMeansModel *model, const Matrix *X_new) {
    if (!model->trained) {
        fprintf(stderr, "Error in X");
        exit(EXIT_FAILURE);
    }

    Matrix predictions = Matrix_zeros(X_new->rows, 1);
    for (int i = 0; i < X_new->rows; ++i) {
        predictions.data[i][0] = find_nearest_centroid(X_new->data[i], &model->cluster_means);
    }

    return predictions;
}

void KMeans_free(KMeansModel *model) {
    Matrix_free(model->data);
    Matrix_free(model->cluster_means);
    Matrix_free(model->labels);
}
