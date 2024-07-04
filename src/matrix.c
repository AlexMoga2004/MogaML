#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int MATRIX_MAX_ITER = 1000;		///< Maximum number of iterations for any numerical method on Matrices
const double MATRIX_TOLERANCE = 1e-6;	///< Maximum margin of error for floating point arithmetic on Matrices

/**
 * @brief Macro to output message to error stream
 */
#define ERROR(fmt, ...) \
        do { \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
        } while (0)

/**
 * @brief Draw a random value from the Standard Normal Distribution using the Box-Muller transformation
 * 
 * @return Value sampled from N(0,1)
 */
static double standard_normal() {
    static int haveSpare = 0;
    static double rand1, rand2;

    if (haveSpare) {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;
    rand1 = rand() / ((double) RAND_MAX);
    if (rand1 < 1e-100) {
        rand1 = 1e-100;
    }
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * 2 * M_PI;

    return sqrt(rand1) * cos(rand2);
}

/** 
 * @brief Frees the memory allocated for the Matrix
 * 
 * @param mat Matrix to be freed 
 * @note Subsequent accessing of Matrix data causes segfault
 */
void Matrix_free(Matrix mat) { 
	for (int i = 0; i < mat.rows; ++i) { 
		free(mat.data[i]);
	} 
	free(mat.data);
} 

/**
 * @brief Sets all entries of the Matrix to 0
 * 
 * @param mat Pointer to Matrix to be reset
 */
void Matrix_reset(Matrix *mat) {
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			mat->data[i][j] = 0;
		}
	}
}

/**
 * @brief Prints the contents of the mAtrix to stdout, formatted with newlines
 * 
 * @param mat Pointer to Matrix to be displayed
 */
void Matrix_display(const Matrix *mat) { 
	for (int i = 0; i < mat->rows; ++i) { 
		for (int j = 0; j < mat->cols; ++j) { 
			printf("%f ", mat->data[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * @brief Checks if two Matrices are pairwise (exactly) equal
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @return true if the matrices have the same dimension, and have equal entries
 * @return false otherwise
 */
bool Matrix_equal(const Matrix *mat1, const Matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return false;
    }
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            if (fabs(mat1->data[i][j] - mat2->data[i][j]) > MATRIX_TOLERANCE) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if two Matrices are pairwise equal within a given tolerance
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @param tolerance Maximum difference between each element 
 * @return true if the matrices have the same dimension, and the entries are within the tolerance of each other
 * @return false otherwise
 */
bool Matrix_approx_equal(const Matrix *mat1, const Matrix *mat2, double tolerance) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return false;
    }
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            if (fabs(mat1->data[i][j] - mat2->data[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Computes the determinant of a square Matrix
 * 
 * @param mat Pointer to the Matrix
 * @return double Value of the determinant
 * @throw Exception for non-square Matrices
 */
double Matrix_det(const Matrix *mat) { 
	if (mat->rows != mat->cols) {
		ERROR("Error in Matrix_det, dimension mismatch!\n");
		exit(EXIT_FAILURE);
	}

	if (mat->rows == 1) return mat->data[0][0]; 

	double result = 0.0;
	for (int i = 0; i < mat->rows; ++i) {
		Matrix minor = Matrix_minor(mat, i, 0);
		double cofactor = (i % 2 == 0) ? 1 : -1;
		result += cofactor * mat->data[i][0] * Matrix_det(&minor);
	}

	return result;
}

/**
 * @brief Computes the sum of the diagonal entries of a square Matrix
 * 
 * @param mat Pointer to the Matrix
 * @returns double Value of the trace
 * @throw Exception for non-square Matrices
 */
double Matrix_trace(const Matrix *mat) {
	if (mat->rows != mat->cols) {
		ERROR("error in Matrix_trace, non-square matrix!\n");
		exit(EXIT_FAILURE);
	} 

	double result = 0.0;

	for (int i = 0; i < mat->rows; ++i) {
		result += mat->data[i][i];
	}

	return result;
}

/**
 * @brief Computes the l2-norm of a Matrix
 * 
 * @param mat Pointer to the Matrix
 * @return double value of the l2-norm
 */
double Matrix_norm(const Matrix *mat) {
    double norm = 0.0;
    for (int i = 0; i < mat->rows; ++i) {
        for (int j = 0; j < mat->cols; ++j) {
            norm += mat->data[i][j] * mat->data[i][j];
        }
    }
    return sqrt(norm);
}

/**
 * @brief Computes the frobenius norm of a Matrix
 * 
 * @param mat Pointer to the Matrix
 * @return double value of the frobenius norm
 */
double Matrix_frobenius_norm(const Matrix *mat) {
	Matrix matT = Matrix_transpose(mat);
	Matrix mat_matT = Matrix_multiply(mat, &matT);
	return sqrt (
		Matrix_trace(&mat_matT)
	);
}

/**
 * @brief Computes the l-k norm of a Vector
 * 
 * @param mat Pointer to the Vector (embedded in a Matrix)
 * @param k Non-negative integer to adjust norm type
 * @return double l-k norm
 */
double Vector_norm(const Matrix *mat, unsigned int k) {
	if (mat->cols != 1) {
		ERROR("Error in Vector_norm, not a vector!");
		exit(EXIT_FAILURE);
	}

	double sum = 0.0;
	for (int i = 0; i < mat->rows; ++i) {
		sum += fabs(pow(mat->data[i][0], k));
	}

	return pow(sum, 1/(double)k);
}

/**
 * @brief Computes the largest value in a row vector
 * 
 * @param mat Pointer to the Vector (embedded in a Matrix)
 * @return double Value of the largest element in the Vector
 * @throw Exception for Matrices that are not row vectors (Must have dimension (Nx1))
 */
double Vector_max(const Matrix *mat) {
	if (mat->cols != 1) {
		ERROR("Error in Vector_max_index, not a vector!");
		exit(EXIT_FAILURE);
	}

	double max = mat->data[0][0];

	for (int i = 0; i < mat->rows; ++i) {
		if (mat->data[i][0] > max) {
			max = mat->data[i][0];
		}
	}

	return max;
}

/**
 * @brief Computes the index of the largest value in a row vector
 * 
 * @param mat Pointer to the Vector (embedded in a Matrix)
 * @return int Index of the largest element in the Vector
 * @throw Exception for Matrices that are not row vectors (Must have dimension (Nx1))
 */
int Vector_max_index(const Matrix *mat) {
	if (mat->cols != 1) {
		ERROR("Error in Vector_max_index, not a vector!");
		exit(EXIT_FAILURE);
	}

	double max = mat->data[0][0];
	double index = 0;

	for (int i = 0; i < mat->rows; ++i) {
		if (mat->data[i][0] > max) {
			index = i;
			max = mat->data[i][0];
		}
	}

	return index;
}

/**
 * @brief Initialises new Matrix with entries set to 0
 * 
 * @param rows Number of rows in the Matrix
 * @param cols Number of columns in the Matrix
 * @return Matrix 
 */
Matrix Matrix_zeros(int rows, int cols) { 
	if (rows <= 0 || cols <= 0) { 
		ERROR("error in Matrix_zeros, non-positive dimension(s)!\n");
		exit(EXIT_FAILURE);
	}

	Matrix mat; 
	mat.rows = rows;
	mat.cols = cols;
	mat.data = (double **)malloc(rows * sizeof(double *));

	for (int i = 0; i < mat.rows; ++i) {
		mat.data[i] = (double *)malloc(cols * sizeof(double *));
	}

	return mat;
}

/**
 * @brief Initialises an Identity matrix (square matrix with diagonal entries 1)
 * 
 * @param size Number of rows and columns
 * @return Matrix 
 */
Matrix Matrix_identity(int size) { 
	Matrix mat = Matrix_zeros(size, size);
	for (int i = 0; i < size; ++i) {
		mat.data[i][i] = 1;
	}

	return mat;
}

/**
 * @brief Initialises a Matrix given a 2D array of elements
 * 
 * @param rows Number of rows in the Matrix
 * @param cols Number of columns in the Matrix
 * @param data 2D Array, containing the rows of the Matrix
 * @return Matrix 
 */
Matrix Matrix_from_array(int rows, int cols, double *data) {
    Matrix matrix = Matrix_zeros(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix.data[i][j] = data[i * cols + j];
        }
    }
    
    return matrix;
}

/**
 * @brief Scales a matrix by a constant factor
 * 
 * @param c Factor
 * @param mat Pointer to original Matrix
 * @return New Matrix
 */
Matrix Matrix_scale(double c, const Matrix *mat) {
	Matrix result = Matrix_zeros(mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; ++i){
		for (int j = 0; j < mat->cols; ++j){
			result.data[i][j] = c * mat->data[i][j];
		}
	}

	return result;
}

/**
 * @brief Sums the entries in two Matrices element-wise
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @return Result of the Matrix addition
 * @throw Exception when the Matrices have different dimensions
 */
Matrix Matrix_add(const Matrix *mat1, const Matrix *mat2) {
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
		ERROR("Error in Matrix_add, dimension mismatch!\n");
		exit(EXIT_FAILURE);
	}

	Matrix result = Matrix_zeros(mat1->rows, mat1->cols);

	for (int i = 0; i < mat1->rows; ++i) {
		for (int j = 0; j < mat1->cols; ++j) {
			result.data[i][j] = mat1->data[i][j] + mat2->data[i][j];
		}
	}

	return result;
}

/**
 * @brief Subtracts the entries in two Matrices element-wise
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @return Result of the Matrix subtraction
 * @throw Exception when the Matrices have different dimensions
 */
Matrix Matrix_sub(const Matrix *mat1, const Matrix *mat2) {
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
		ERROR("Error in Matrix_add, dimension mismatch!\n");
		exit(EXIT_FAILURE);
	}

	Matrix result = Matrix_zeros(mat1->rows, mat1->cols);

	for (int i = 0; i < mat1->rows; ++i) {
		for (int j = 0; j < mat1->cols; ++j) {
			result.data[i][j] = mat1->data[i][j] - mat2->data[i][j];
		}
	}

	return result;
}

/**
 * @brief Multiplies two Matrices
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @return Result of Matrix multiplication
 * @throw Exception requires mat1->cols == mat2->rows
 */
Matrix Matrix_multiply(const Matrix *mat1, const Matrix *mat2) { 
	// Treat 1x1 matrix as a constant
	if (mat1->rows == 1 && mat1-> cols == 1) {
		return Matrix_scale(mat1->data[0][0], mat2);
	} else if (mat2->rows == 1 && mat2->cols == 1) {
		return Matrix_scale(mat2->data[0][0], mat1);
	}

	if (mat1->cols != mat2->rows) { 
		ERROR("Error in Matrix_multiply, dimension mismatch\n");
		exit(EXIT_FAILURE);
	} 

	Matrix result = Matrix_zeros(mat1->rows, mat2->cols);

	for (int i = 0; i < mat1->rows; ++i){
		for (int j = 0; j < mat2->cols; ++j){
			for (int k = 0; k < mat1->cols; ++k){
				result.data[i][j] += mat1->data[i][k] * mat2->data[k][j];
			}
		}
	}

	return result;
}

/**
 * @brief Computes the minor of a Matrix by removing a row and a column
 * 
 * @param mat Pointer to original Matrix
 * @param row Index of row to remove
 * @param col Index of column to remove
 * @return Remaining Matrix minor
 * @throw Exception requires 0 <= row < Matrix.rows && 0 <= col < Matrix.cols
 */
Matrix Matrix_minor(const Matrix *mat, int row, int col) {
	if (mat->rows <= 1 || mat->cols <= 1) {
		ERROR("Error in Matrix_minor, cannot take minor for dimension <= 1!\n");
		exit(EXIT_FAILURE);
	}

	Matrix result = Matrix_zeros(mat->rows - 1, mat->cols - 1);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			result.data[i][j] = mat->data[i][j];
		}
		for (int j = col+1; j < mat->cols; ++j) {
			result.data[i][j-1] = mat->data[i][j];
		}
	}

	for (int i = row+1; i < mat->rows; ++i) {
		for (int j = 0; j < col; ++j) {
			result.data[i-1][j] = mat->data[i][j];
		}
		for (int j = col+1; j < mat->cols; ++j) {
			result.data[i-1][j-1] = mat->data[i][j];
		}
	}

	return result;
}

/**
 * @brief Get a single row from a matrix
 * 
 * @param mat Pointer to original Matrix
 * @param row Index of row to fetch
 * @return Row vector (in Matrix form) containing the desired row 
 * @throw Exception requires 0 <= row < Matrix.rows
 */
Matrix Matrix_row(const Matrix *mat, int row_index) {
    if (row_index < 0 || row_index >= mat->rows) {
        ERROR("Error in Matrix_row: row index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    Matrix row = Matrix_zeros(1, mat->cols);
    for (int j = 0; j < mat->cols; ++j) {
        row.data[0][j] = mat->data[row_index][j];
    }
    return row;
}

/**
 * @brief Get a single column from a matrix
 * 
 * @param mat Pointer to original Matrix
 * @param col Index of column to fetch
 * @return Column vector (in Matrix form) containing the desired column 
 * @throw Exception requires 0 <= col < Matrix.cols
 */
Matrix Matrix_col(const Matrix *mat, int col_index) {
    if (col_index < 0 || col_index >= mat->cols) {
        ERROR("Error in Matrix_row: row index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    Matrix col = Matrix_zeros(1, mat->cols);
    for (int j = 0; j < mat->rows; ++j) {
        col.data[j][0] = mat->data[j][col_index];
    }
    return col;
}

/**
 * @brief Get rows within the specified range
 * 
 * @param mat Original matrix
 * @param start Index to start fetching from (inclusive)
 * @param end Index to stop fetching (exclusive)
 * @return Sub-matrix containing specified rows
 * @throw Exception requires 0 <= start < end <= matrix.rows
 */
Matrix Matrix_slice_rows(const Matrix *mat, int start, int end) {
    int num_rows = end - start;
    Matrix slice = Matrix_zeros(num_rows, mat->cols);
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < mat->cols; ++j) {
            slice.data[i - start][j] = mat->data[i][j];
        }
    }
    return slice;
}

/**
 * @brief Get a Matrix that contains a subset of the rows and columns
 * 
 * @param mat Pointer to original Matrix
 * @param start_row Index to start fetching rows from (inclusive)
 * @param end_row Index to stop fetching rows from (exclusive)
 * @param start_col Index to start fetching cols from (inclusive)
 * @param end_col Index to stop fetching cols from (exclusive)
 * @return Matrix containing the specified rows and columns
 * @throw Exception requires 0 <= start_row < end_row <= mat->rows && 0 <= start_col < end_col <= mat->cols
 */
Matrix Matrix_submatrix(const Matrix *mat, int start_row, int end_row, int start_col, int end_col) {
    if (start_row < 0 || start_row >= mat->rows || end_row <= start_row || end_row > mat->rows ||
        start_col < 0 || start_col >= mat->cols || end_col <= start_col || end_col > mat->cols) {
        ERROR("Error in Matrix_submatrix: invalid submatrix indices\n");
        exit(EXIT_FAILURE);
    }

    int sub_rows = end_row - start_row;
    int sub_cols = end_col - start_col;

    Matrix submat = Matrix_zeros(sub_rows, sub_cols);

    for (int i = 0; i < sub_rows; ++i) {
        for (int j = 0; j < sub_cols; ++j) {
            submat.data[i][j] = mat->data[start_row + i][start_col + j];
        }
    }

    return submat;
}


/**
 * @brief Flip the rows and columns of a Matrix
 * 
 * @param mat Pointer to original matrix
 * @return Transposed Matrix
 */
Matrix Matrix_transpose(const Matrix *mat) {
	Matrix result = Matrix_zeros(mat->cols, mat->rows);
	for (int i = 0; i < mat->rows; ++i){
		for (int j = 0; j < mat->cols; ++j){
			result.data[j][i] = mat->data[i][j];
		}
	}

	return result;
}

/**
 * @brief Create a new matrix with identical entries
 * 
 * @param mat Pointer to originial Matrix
 * @return Identical Matrix
 */
Matrix Matrix_clone(const Matrix *mat) {
	Matrix result = Matrix_zeros(mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			result.data[i][j] = mat->data[i][j];
		}
	}
	return result;
}

/**
 * @brief Compute the inverse of a square Matrix
 * 
 * @param mat Pointer to original Matrix
 * @return Matrix 
 * @throw Exception for non-square Matrices, and for singular Matrices
 */
Matrix Matrix_inverse(const Matrix *mat) {
	if (fabs(Matrix_det(mat)) < MATRIX_TOLERANCE) {
		ERROR("Error in Matrix_inverse, singular matrix!\n");
		exit(EXIT_FAILURE);
	}

	SVDResult svd = Matrix_svd(mat);
	Matrix SigmaInverse = Matrix_zeros(svd.Sigma.rows, svd.Sigma.cols);
	for (int i = 0; i < svd.Sigma.rows; ++i) {
		SigmaInverse.data[i][i] = 1 / svd.Sigma.data[i][i];
	}

	Matrix UInverse = Matrix_transpose(&svd.U);
	Matrix VSigma = Matrix_multiply(&svd.V, &SigmaInverse);

	return Matrix_multiply(&VSigma, &UInverse);
}

/**
 * @brief Return the solution to the equation Ax=b
 * 
 * @param A Pointer to Matrix A
 * @param b Pointer to column Vector b (embedded as a Matrix)
 * @return Column vector x as a Matrix, the solution to the equation
 * @throw Exception for incompatible dimensions, or singular Matrices A
 */
Matrix Matrix_solve(const Matrix *A, const Matrix *b) {
	if (A->rows != A->cols) {
		ERROR("Error in Matrix_solve, not a square matrix!\n");
		exit(EXIT_FAILURE);
	}

	if (b->cols != 1) {
		ERROR("Error in Matrix_solve, b must be a vector\n!");
		exit(EXIT_FAILURE);
	}

	if (fabs(Matrix_det(A)) < MATRIX_TOLERANCE) {
		ERROR("Error in Matrix_solve, A must be non-singular!\n");
		exit(EXIT_FAILURE);
	}

	Matrix AI = Matrix_inverse(A);
	return Matrix_multiply(&AI, b);
}

/**
 * @brief Compute the Singular Value Decomposition (SVD) of a Matrix
 * 
 * @param mat Pointer to Matrix
 * @return SVDResult containing orthonormal Matrices U, V, and diagonal Matrix Sigma with entries in decreasing order
 */
SVDResult Matrix_svd(const Matrix *mat) {
	int k = (mat->rows < mat->cols) ? mat->rows : mat->cols;
    SVDResult result;

    // Initialize U, Sigma, V matrices
    result.U = Matrix_zeros(mat->rows, k);
    result.Sigma = Matrix_zeros(k, k);
    result.V = Matrix_zeros(mat->cols, k);

    Matrix matT = Matrix_transpose(mat);
    Matrix matTmat = Matrix_multiply(&matT, mat);

    for (int s = 0; s < k; ++s) {
        // Initialize random vector x
        Matrix x = Matrix_zeros(mat->cols, 1);
        for (int i = 0; i < mat->cols; ++i) {
            x.data[i][0] = standard_normal();
        }

        Matrix y_old = Matrix_clone(&x);
        Matrix y_new;

        // Power iteration with orthogonalization to find eigenvector
        for (int iter = 0; iter < MATRIX_MAX_ITER; ++iter) {
            y_new = Matrix_multiply(&matTmat, &y_old);

            // Orthogonalize y_new against all previously found singular vectors
            for (int j = 0; j < s; ++j) {
                double dot_product = 0;
                for (int l = 0; l < mat->cols; ++l) {
                    dot_product += y_new.data[l][0] * result.V.data[l][j];
                }
                for (int l = 0; l < mat->cols; ++l) {
                    y_new.data[l][0] -= dot_product * result.V.data[l][j];
                }
            }

            double magnitude = Vector_norm(&y_new, 2);
            y_new = Matrix_scale(1 / magnitude, &y_new);

            Matrix diff = Matrix_sub(&y_new, &y_old);
            if (Vector_norm(&diff, 2) < MATRIX_TOLERANCE) {
                Matrix_free(diff);
                break;
            }

            Matrix_free(y_old);
            y_old = y_new;
        }

        for (int i = 0; i < mat->cols; ++i) {
            result.V.data[i][s] = y_new.data[i][0];
        }

        Matrix u_s = Matrix_multiply(mat, &y_new);
        double sigma = Vector_norm(&u_s, 2);
        result.Sigma.data[s][s] = sigma;

        u_s = Matrix_scale(1 / sigma, &u_s);
        for (int i = 0; i < mat->rows; ++i) {
            result.U.data[i][s] = u_s.data[i][0];
        }

        Matrix_free(y_old);
        Matrix_free(u_s);
    }

    Matrix_free(matT);
    Matrix_free(matTmat);

    return result;
}
