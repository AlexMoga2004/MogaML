#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int MATRIX_MAX_ITER = 1000;
const double MATRIX_TOLERANCE = 1e-6;

// Sample from N(0,1) using Box-Muller transformation
double standard_normal() {
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

void Matrix_free(Matrix mat) { 
	for (int i = 0; i < mat.rows; ++i) { 
		free(mat.data[i]);
	} 
	free(mat.data);
} 

void Matrix_display(const Matrix *mat) { 
	for (int i = 0; i < mat->rows; ++i) { 
		for (int j = 0; j < mat->cols; ++j) { 
			printf("%f ", mat->data[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int Matrix_equal(const Matrix *mat1, const Matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return 0;
    }
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            if (fabs(mat1->data[i][j] - mat2->data[i][j]) > MATRIX_TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

int Matrix_approx_equal(const Matrix *mat1, const Matrix *mat2, double tolerance) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return 0;
    }
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            if (fabs(mat1->data[i][j] - mat2->data[i][j]) > tolerance) {
                return 0;
            }
        }
    }
    return 1;
}

double Matrix_det(const Matrix *mat) { 
	if (mat->rows != mat->cols) {
		fprintf(stderr, "Error in Matrix_det, dimension mismatch!\n");
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

double Matrix_trace(const Matrix *mat) {
	if (mat->rows != mat->cols) {
		fprintf(stderr, "error in Matrix_trace, non-square matrix!\n");
		exit(EXIT_FAILURE);
	} 

	double result = 0.0;

	for (int i = 0; i < mat->rows; ++i) {
		result += mat->data[i][i];
	}

	return result;
}

double Matrix_norm(const Matrix *mat) {
    double norm = 0.0;
    for (int i = 0; i < mat->rows; ++i) {
        for (int j = 0; j < mat->cols; ++j) {
            norm += mat->data[i][j] * mat->data[i][j];
        }
    }
    return sqrt(norm);
}

double Matrix_frobenius_norm(const Matrix *mat) {
	Matrix matT = Matrix_transpose(mat);
	Matrix mat_matT = Matrix_multiply(mat, &matT);
	return sqrt (
		Matrix_trace(&mat_matT)
	);
}

double Vector_norm(const Matrix *mat, double l) {
	if (mat->cols != 1) {
		fprintf(stderr, "Error in Vector_norm, not a vector!");
		exit(EXIT_FAILURE);
	}

	double sum = 0.0;
	for (int i = 0; i < mat->rows; ++i) {
		sum += fabs(pow(mat->data[i][0], l));
	}

	return pow(sum, 1/l);
}

Matrix Matrix_zeros(int rows, int cols) { 
	if (rows <= 0 || cols <= 0) { 
		fprintf(stderr, "error in Matrix_zeros, non-positive dimension(s)!\n");
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

Matrix Matrix_identity(int size) { 
	Matrix mat = Matrix_zeros(size, size);
	for (int i = 0; i < size; ++i) {
		mat.data[i][i] = 1;
	}

	return mat;
}

Matrix Matrix_scale(double c, const Matrix *mat) {
	Matrix result = Matrix_zeros(mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; ++i){
		for (int j = 0; j < mat->cols; ++j){
			result.data[i][j] = c * mat->data[i][j];
		}
	}

	return result;
}

Matrix Matrix_add(const Matrix *mat1, const Matrix *mat2) {
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
		fprintf(stderr, "Error in Matrix_add, dimension mismatch!\n");
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

Matrix Matrix_sub(const Matrix *mat1, const Matrix *mat2) {
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
		fprintf(stderr, "Error in Matrix_add, dimension mismatch!\n");
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

Matrix Matrix_multiply(const Matrix *mat1, const Matrix *mat2) { 
	// Treat 1x1 matrix as a constant
	if (mat1->rows == 1 && mat1-> cols == 1) {
		return Matrix_scale(mat1->data[0][0], mat2);
	} else if (mat2->rows == 1 && mat2->cols == 1) {
		return Matrix_scale(mat2->data[0][0], mat1);
	}

	if (mat1->cols != mat2->rows) { 
		fprintf(stderr, "Error in Matrix_multiply, dimension mismatch\n");
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

Matrix Matrix_minor(const Matrix *mat, int row, int col) {
	if (mat->rows <= 1 || mat->cols <= 1) {
		fprintf(stderr, "Error in Matrix_minor, cannot take minor for dimension <= 1!\n");
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

Matrix Matrix_row(const Matrix *mat, int row_index) {
    if (row_index < 0 || row_index >= mat->rows) {
        fprintf(stderr, "Error in Matrix_row: row index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    Matrix row = Matrix_zeros(1, mat->cols);
    for (int j = 0; j < mat->cols; ++j) {
        row.data[0][j] = mat->data[row_index][j];
    }
    return row;
}

Matrix Matrix_col(const Matrix *mat, int col_index) {
    if (col_index < 0 || col_index >= mat->cols) {
        fprintf(stderr, "Error in Matrix_row: row index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    Matrix col = Matrix_zeros(1, mat->cols);
    for (int j = 0; j < mat->rows; ++j) {
        col.data[j][0] = mat->data[j][col_index];
    }
    return col;
}

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

Matrix Matrix_transpose(const Matrix *mat) {
	Matrix result = Matrix_zeros(mat->cols, mat->rows);
	for (int i = 0; i < mat->rows; ++i){
		for (int j = 0; j < mat->cols; ++j){
			result.data[j][i] = mat->data[i][j];
		}
	}

	return result;
}

Matrix Matrix_clone(const Matrix *mat) {
	Matrix result = Matrix_zeros(mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			result.data[i][j] = mat->data[i][j];
		}
	}
	return result;
}

Matrix Matrix_inverse(const Matrix *mat) {
	if (fabs(Matrix_det(mat)) < MATRIX_TOLERANCE) {
		fprintf(stderr, "Error in Matrix_inverse, singular matrix!\n");
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

// Solve the system Ax=b, requires A square, x,b vectors
Matrix Matrix_solve(const Matrix *A, const Matrix *b) {
	if (A->rows != A->cols) {
		fprintf(stderr, "Error in Matrix_solve, not a square matrix!\n");
		exit(EXIT_FAILURE);
	}

	if (b->cols != 1) {
		fprintf(stderr, "Error in Matrix_solve, b must be a vector\n!");
		exit(EXIT_FAILURE);
	}

	if (fabs(Matrix_det(A)) < MATRIX_TOLERANCE) {
		fprintf(stderr, "Error in Matrix_solve, A must be non-singular!\n");
		exit(EXIT_FAILURE);
	}

	Matrix AI = Matrix_inverse(A);
	return Matrix_multiply(&AI, b);
}

SVDResult Matrix_svd(const Matrix *mat) {
	int k = (mat->rows < mat->cols) ? mat->rows : mat->cols;
    SVDResult result;

    // Initialize U, Sigma, V matrices
    result.U = Matrix_zeros(mat->rows, k);
    result.Sigma = Matrix_zeros(k, k);
    result.V = Matrix_zeros(mat->cols, k);

    Matrix matT = Matrix_transpose(mat);
    Matrix matTmat = Matrix_multiply(&matT, mat);

    // for (int s = 0; s < 1; ++s) {
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
