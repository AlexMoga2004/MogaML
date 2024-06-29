#pragma once
#include "matrix.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>

Matrix PCA_Reduce(const Matrix *X, unsigned int target_rank);
Matrix ChiSquared_Reduce(const Matrix *X, const Matrix *y, unsigned int target_features);
