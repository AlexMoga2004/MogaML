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
 * @param mat Pointer to Matrix to be reset
 */
void Matrix_reset(Matrix *mat);

/**
 * @brief Prints the contents of the mAtrix to stdout, formatted with newlines
 * 
 * @param mat Pointer to Matrix to be displayed
 */
void Matrix_display(const Matrix *mat);

/**
 * @brief Checks if two Matrices are pairwise (exactly) equal
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @return true if the matrices have the same dimension, and have equal entries
 * @return false otherwise
 */
bool Matrix_equal(const Matrix *mat1, const Matrix *mat2);

/**
 * @brief Checks if two Matrices are pairwise equal within a given tolerance
 * 
 * @param mat1 Pointer to first Matrix
 * @param mat2 Pointer to second Matrix
 * @param tolerance Maximum difference between each element 
 * @return true if the matrices have the same dimension, and the entries are within the tolerance of each other
 * @return false otherwise
 */
bool Matrix_approx_equal(const Matrix *mat1, const Matrix *mat2, double tolerance);

/**
 * @brief Computes the determinant of a square Matrix
 * 
 * @param mat Pointer to the Matrix
 * @return double Value of the determinant
 * @throw Exception for non-square Matrices
 */
double Matrix_det(const Matrix *mat);

/**
 * @brief Computes the sum of the diagonal entries of a square Matrix
 * 
 * @param mat Pointer to the Matrix
 * @returns double Value of the trace
 * @throw Exception for non-square Matrices
 */
double Matrix_trace(const Matrix *mat);

/**
 * @brief Computes the l2-norm of a Matrix
 * 
 * @param mat Pointer to the Matrix
 * @return double value of the l2-norm
 */
double Matrix_norm(const Matrix *mat);

/**
 * @brief Computes the frobenius norm of a Matrix
 * 
 * @param mat Pointer to the Matrix
 * @return double value of the frobenius norm
 */
double Matrix_frobenius_norm(const Matrix *mat);

/**
 * @brief Computes the l-k norm of a Vector
 * 
 * @param mat Pointer to the Vector (embedded in a Matrix)
 * @param k Non-negative integer to adjust norm type
 * @return double l-k norm
 */
double Vector_norm(const Matrix *mat, unsigned int k);

/**
 * @brief Computes the largest value in a row vector
 * 
 * @param mat Pointer to the Vector (embedded in a Matrix)
 * @return double Value of the largest element in the Vector
 * @throw Exception for Matrices that are not row vectors (Must have dimension (Nx1))
 */
double Vector_max(const Matrix *mat);

/**
 * @brief Computes the index of the largest value in a row vector
 * 
 * @param mat Pointer to the Vector (embedded in a Matrix)
 * @return int Index of the largest element in the Vector
 * @throw Exception for Matrices that are not row vectors (Must have dimension (Nx1))
 */
int Vector_max_index(const Matrix *mat);

/**
 * @brief 
 * 
 * @param rows 
 * @param cols 
 * @return Matrix 
 */
Matrix Matrix_zeros(int rows, int cols);
Matrix Matrix_identity(int size);
Matrix Matrix_from_array(int rows, int cols, double *data);
Matrix Matrix_scale(double c, const Matrix *mat);
Matrix Matrix_add(const Matrix *mat1, const Matrix *mat2);
Matrix Matrix_sub(const Matrix *mat1, const Matrix *mat2);
Matrix Matrix_multiply(const Matrix *mat1, const Matrix *mat2);
Matrix Matrix_minor(const Matrix *mat, int row, int col);
Matrix Matrix_row(const Matrix *mat, int row);
Matrix Matrix_col(const Matrix *mat, int col);
Matrix Matrix_slice_rows(const Matrix *mat, int start, int end);
Matrix Matrix_submatrix(const Matrix *mat, int start_row, int end_row, int start_col, int end_col);
Matrix Matrix_transpose(const Matrix *mat);
Matrix Matrix_clone(const Matrix *mat);
Matrix Matrix_inverse(const Matrix *mat);
Matrix Matrix_solve(const Matrix *A, const Matrix *b);

SVDResult Matrix_svd(const Matrix *mat);
