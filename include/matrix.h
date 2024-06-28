#pragma once

typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

typedef struct {
	Matrix U;
	Matrix Sigma;
	Matrix V;
} SVDResult;

void Matrix_free(Matrix mat);
void Matrix_reset(Matrix *mat);
void Matrix_display(const Matrix *mat);

int Matrix_equal(const Matrix *mat1, const Matrix *mat2);
int Matrix_approx_equal(const Matrix *mat1, const Matrix *mat2, double tolerance);

double Matrix_det(const Matrix *mat);
double Matrix_trace(const Matrix *mat);
double Matrix_norm(const Matrix *mat);
double Matrix_frobenius_norm(const Matrix *mat);

double Vector_max(const Matrix *mat);
double Vector_max_index(const Matrix *mat);
double Vector_norm(const Matrix *mat, double l);

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
Matrix Matrix_transpose(const Matrix *mat);
Matrix Matrix_clone(const Matrix *mat);
Matrix Matrix_inverse(const Matrix *mat);
Matrix Matrix_solve(const Matrix *A, const Matrix *b);

SVDResult Matrix_svd(const Matrix *mat);
