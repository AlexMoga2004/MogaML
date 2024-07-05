#pragma once
#include <stdbool.h>

/**
 * @brief Structure representing a Matrix
 */
typedef struct {
    int rows;       ///< Number of rows
    int cols;       ///< Number of columns
    double** data;  ///< 2D array of data
} Matrix;

/**
 * @brief Structure representing the result after computing Singular Value Decompositions
 */
typedef struct {
	Matrix U;       ///< Orthonormal Matrix U
	Matrix Sigma;   ///< Diagonal Matrix Sigma, with entries in decreasing order
	Matrix V;       ///< Orthonormal Matrix V
} SVDResult;


/** 
 * @brief Frees the memory allocated for the Matrix
 * 
 * @param mat Matrix to be freed 
 * @note Subsequent accessing of Matrix data causes segfault
 */
void Matrix_free(Matrix mat);

/**
 * @brief Sets all entries of the Matrix to 0
 * 
 * @param mat Matrix to be reset
 */
void Matrix_reset(Matrix *mat);

/**
 * @brief Prints the contents of the mAtrix to stdout, formatted with newlines
 * 
 * @param mat Matrix to be displayed
 */
void Matrix_display(const Matrix *mat);

/**
 * @brief Checks if two Matrices are pairwise (exactly) equal
 * 
 * @param mat1 first Matrix
 * @param mat2 second Matrix
 * @return true if the matrices have the same dimension, and have equal entries
 * @return false otherwise
 */
bool Matrix_equal(const Matrix *mat1, const Matrix *mat2);

/**
 * @brief Checks if two Matrices are pairwise equal within a given tolerance
 * 
 * @param mat1 first Matrix
 * @param mat2 second Matrix
 * @param tolerance Maximum difference between each element 
 * @return true if the matrices have the same dimension, and the entries are within the tolerance of each other
 * @return false otherwise
 */
bool Matrix_approx_equal(const Matrix *mat1, const Matrix *mat2, double tolerance);

/**
 * @brief Computes the determinant of a square Matrix
 * 
 * @param mat the Matrix
 * @return double Value of the determinant
 * @throw Exception for non-square Matrices
 */
double Matrix_det(const Matrix *mat);

/**
 * @brief Computes the sum of the diagonal entries of a square Matrix
 * 
 * @param mat the Matrix
 * @returns double Value of the trace
 * @throw Exception for non-square Matrices
 */
double Matrix_trace(const Matrix *mat);

/**
 * @brief Computes the l2-norm of a Matrix
 * 
 * @param mat the Matrix
 * @return double value of the l2-norm
 */
double Matrix_norm(const Matrix *mat);

/**
 * @brief Computes the frobenius norm of a Matrix
 * 
 * @param mat the Matrix
 * @return double value of the frobenius norm
 */
double Matrix_frobenius_norm(const Matrix *mat);

/**
 * @brief Computes the l-k norm of a Vector
 * 
 * @param mat the Vector (embedded in a Matrix)
 * @param k Non-negative integer to adjust norm type
 * @return double l-k norm
 */
double Vector_norm(const Matrix *mat, unsigned int k);

/**
 * @brief Computes the largest value in a column vector
 * 
 * @param mat the Vector (embedded in a Matrix)
 * @return double Value of the largest element in the Vector
 * @throw Exception for Matrices that are not column vectors (Must have dimension (Nx1))
 */
double Vector_max(const Matrix *mat);

/**
 * @brief Computes the index of the largest value in a column vector
 * 
 * @param mat the Vector (embedded in a Matrix)
 * @return int Index of the largest element in the Vector
 * @throw Exception for Matrices that are not column vectors (Must have dimension (Nx1))
 */
int Vector_max_index(const Matrix *mat);

/**
 * @brief Initialises new Matrix with entries set to 0
 * 
 * @param rows Number of rows in the Matrix
 * @param cols Number of columns in the Matrix
 * @return Matrix 
 */
Matrix Matrix_zeros(int rows, int cols);

/**
 * @brief Initialises an Identity matrix (square matrix with diagonal entries 1)
 * 
 * @param size Number of rows and columns
 * @return Matrix 
 */
Matrix Matrix_identity(int size);

/**
 * @brief Initialises a Matrix given a 2D array of elements
 * 
 * @param rows Number of rows in the Matrix
 * @param cols Number of columns in the Matrix
 * @param data 2D Array, containing the rows of the Matrix
 * @return Matrix 
 */
Matrix Matrix_from_array(int rows, int cols, double *data);

/**
 * @brief Scales a matrix by a constant factor
 * 
 * @param c Factor
 * @param mat original Matrix
 * @return New Matrix
 */
Matrix Matrix_scale(double c, const Matrix *mat);

/**
 * @brief Sums the entries in two Matrices element-wise
 * 
 * @param mat1 first Matrix
 * @param mat2 second Matrix
 * @return Result of the Matrix addition
 * @throw Exception when the Matrices have different dimensions
 */
Matrix Matrix_add(const Matrix *mat1, const Matrix *mat2);

/**
 * @brief Subtracts the entries in two Matrices element-wise
 * 
 * @param mat1 first Matrix
 * @param mat2 second Matrix
 * @return Result of the Matrix subtraction
 * @throw Exception when the Matrices have different dimensions
 */
Matrix Matrix_sub(const Matrix *mat1, const Matrix *mat2);

/**
 * @brief Multiplies two Matrices
 * 
 * @param mat1 first Matrix
 * @param mat2 second Matrix
 * @return Result of Matrix multiplication
 * @throw Exception, requires mat1->cols == mat2->rows
 */
Matrix Matrix_multiply(const Matrix *mat1, const Matrix *mat2);

/**
 * @brief Computes the minor of a Matrix by removing a row and a column
 * 
 * @param mat original Matrix
 * @param row Index of row to remove
 * @param col Index of column to remove
 * @return Remaining Matrix minor
 * @throw Exception, requires 0 <= row < Matrix.rows && 0 <= col < Matrix.cols
 */
Matrix Matrix_minor(const Matrix *mat, int row, int col);

/**
 * @brief Get a single row from a matrix
 * 
 * @param mat original Matrix
 * @param row Index of row to fetch
 * @return Row vector (in Matrix form) containing the desired row 
 * @throw Exception, requires 0 <= row < Matrix.rows
 */
Matrix Matrix_row(const Matrix *mat, int row);

/**
 * @brief Get a single column from a matrix
 * 
 * @param mat original Matrix
 * @param col Index of column to fetch
 * @return Column vector (in Matrix form) containing the desired column 
 * @throw Exception, requires 0 <= col < Matrix.cols
 */
Matrix Matrix_col(const Matrix *mat, int col);

/**
 * @brief Get rows within the specified range
 * 
 * @param mat Original matrix
 * @param start Index to start fetching from (inclusive)
 * @param end Index to stop fetching (exclusive)
 * @return Sub-matrix containing specified rows
 * @throw Exception, requires 0 <= start < end <= matrix.rows
 */
Matrix Matrix_slice_rows(const Matrix *mat, int start, int end);

/**
 * @brief Get a Matrix that contains a subset of the rows and columns
 * 
 * @param mat original Matrix
 * @param start_row Index to start fetching rows from (inclusive)
 * @param end_row Index to stop fetching rows from (exclusive)
 * @param start_col Index to start fetching cols from (inclusive)
 * @param end_col Index to stop fetching cols from (exclusive)
 * @return Matrix containing the specified rows and columns
 * @throw Exception, requires 0 <= start_row < end_row <= mat->rows && 0 <= start_col < end_col <= mat->cols
 */
Matrix Matrix_submatrix(const Matrix *mat, int start_row, int end_row, int start_col, int end_col);

/**
 * @brief Flip the rows and columns of a Matrix
 * 
 * @param mat original matrix
 * @return Transposed Matrix
 */
Matrix Matrix_transpose(const Matrix *mat);

/**
 * @brief Create a new matrix with identical entries
 * 
 * @param mat originial Matrix
 * @return Identical Matrix
 */
Matrix Matrix_clone(const Matrix *mat);

/**
 * @brief Compute the inverse of a square Matrix
 * 
 * @param mat original Matrix
 * @return Matrix 
 * @throw Exception for non-square Matrices, and for singular Matrices
 */
Matrix Matrix_inverse(const Matrix *mat);

/**
 * @brief Return the solution to the equation Ax=b
 * 
 * @param A Matrix A
 * @param b column Vector b (embedded as a Matrix)
 * @return Column vector x as a Matrix, the solution to the equation
 * @throw Exception, for incompatible dimensions, or singular Matrices A
 */
Matrix Matrix_solve(const Matrix *A, const Matrix *b);

/**
 * @brief Compute the Singular Value Decomposition (SVD) of a Matrix
 * 
 * @param mat Matrix
 * @return SVDResult containing orthonormal Matrices U, V, and diagonal Matrix Sigma with entries in decreasing order
 */
SVDResult Matrix_svd(const Matrix *mat);
