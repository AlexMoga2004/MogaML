#pragma once
#include "matrix.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>

Matrix Unsupervised_read_csv(const char *filename);

/*                  K-Means                 */
typedef struct {
    int k;
    bool trained;

    Matrix data;
    Matrix cluster_means;
    Matrix labels;
} KMeansModel;

KMeansModel KMeans(unsigned int k, const Matrix *X);

void KMeans_train(KMeansModel *model);
Matrix KMeans_predict(const KMeansModel *model, const Matrix *X_new);
void KMeans_free(KMeansModel *model);
