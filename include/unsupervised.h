#pragma once
#include "matrix.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>

/**
 * @brief Reads a Matrix from a Comma Separated Value (CSV) file
 * 
 * @param filename
 * @return Matrix read data
 */
Matrix Unsupervised_read_csv(const char *filename);

/*                  K-Means                 */
typedef struct {
    int k;
    bool trained;

    Matrix data;
    Matrix cluster_means;
    Matrix labels;
} KMeansModel;

/**
 * @brief Construct a KMeansModel with a set of data
 * 
 * @param k Number of clusters
 * @param X Data
 * @return KMeansModel 
 */
KMeansModel KMeans(unsigned int k, const Matrix *X);

/**
 * @brief Train the model using lloyd's algorithm
 * 
 * @param model Model to be trained
 */
void KMeans_train(KMeansModel *model);

/**
 * @brief Predict the cluster of unseen data using a KMeansModel
 * 
 * @param model 
 * @param X_new 
 * @return Matrix Predicted clusters
 */
Matrix KMeans_predict(const KMeansModel *model, const Matrix *X_new);

/**
 * @brief Free memory allocated to KMeans Model
 * 
 * @param model Model to be freed
 * @note Subsequent accesses to the model may segfault
 */
void KMeans_free(KMeansModel *model);
