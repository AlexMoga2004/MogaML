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
 * @brief Computes a low-rank approximation of a matrix using Principle Component Analysis
 * 
 * @param X Pointer to the Matrix
 * @param target_rank rank of the output Matrix
 * @return Matrix of rank target_rank
 */
Matrix PCA_Reduce(const Matrix *X, unsigned int target_rank);

/**
 * @brief Selects the most relevant input features using the chi-squared statistical test
 * 
 * @param X Pointer to original input data
 * @param y Pointer to output labels
 * @param target_features Number of features to select
 * @return Matrix with target_features input features
 */
Matrix ChiSquared_Reduce(const Matrix *X, const Matrix *y, unsigned int target_features);