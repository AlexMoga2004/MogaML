#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/matrix.h"

const double tolerance = 1e-4;

// Helper function to compare two matrices for equality
int matrix_equal(const Matrix *mat1, const Matrix *mat2) {
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

void test_matrix_zeros() {
    printf("Testing Matrix_zeros...\n");
    Matrix A = Matrix_zeros(2, 3);
    assert(A.rows == 2 && A.cols == 3);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            assert(A.data[i][j] == 0.0);
        }
    }
    Matrix_free(A);
    printf("Matrix_zeros test passed.\n");
}

void test_matrix_identity() {
    printf("Testing Matrix_identity...\n");
    Matrix I = Matrix_identity(3);
    assert(I.rows == 3 && I.cols == 3);
    for (int i = 0; i < I.rows; i++) {
        for (int j = 0; j < I.cols; j++) {
            if (i == j) {
                assert(I.data[i][j] == 1.0);
            } else {
                assert(I.data[i][j] == 0.0);
            }
        }
    }
    Matrix_free(I);
    printf("Matrix_identity test passed.\n");
}

void test_matrix_scale() {
    printf("Testing Matrix_scale...\n");
    Matrix I = Matrix_identity(3);
    Matrix S = Matrix_scale(2.0, &I);
    assert(S.rows == 3 && S.cols == 3);
    for (int i = 0; i < S.rows; i++) {
        for (int j = 0; j < S.cols; j++) {
            assert(S.data[i][j] == 2.0 * I.data[i][j]);
        }
    }
    Matrix_free(I);
    Matrix_free(S);
    printf("Matrix_scale test passed.\n");
}

void test_matrix_multiply() {
    printf("Testing Matrix_multiply...\n");
    Matrix A = Matrix_zeros(2, 2);
    A.data[0][0] = 1; A.data[0][1] = 2;
    A.data[1][0] = 3; A.data[1][1] = 4;

    Matrix B = Matrix_identity(2);
    B.data[0][0] = 2; B.data[1][1] = 2;

    Matrix C = Matrix_multiply(&A, &B);
    assert(C.rows == 2 && C.cols == 2);
    assert(C.data[0][0] == 2.0);
    assert(C.data[0][1] == 4.0);
    assert(C.data[1][0] == 6.0);
    assert(C.data[1][1] == 8.0);

    Matrix_free(A);
    Matrix_free(B);
    Matrix_free(C);
    printf("Matrix_multiply test passed.\n");
}

void test_matrix_transpose() {
    printf("Testing Matrix_transpose...\n");
    Matrix A = Matrix_zeros(2, 3);
    A.data[0][0] = 1; A.data[0][1] = 2; A.data[0][2] = 3;
    A.data[1][0] = 4; A.data[1][1] = 5; A.data[1][2] = 6;

    Matrix T = Matrix_transpose(&A);
    assert(T.rows == 3 && T.cols == 2);
    assert(T.data[0][0] == 1.0);
    assert(T.data[1][0] == 2.0);
    assert(T.data[2][0] == 3.0);
    assert(T.data[0][1] == 4.0);
    assert(T.data[1][1] == 5.0);
    assert(T.data[2][1] == 6.0);

    Matrix_free(A);
    Matrix_free(T);
    printf("Matrix_transpose test passed.\n");
}

void test_matrix_minor() {
    printf("Testing Matrix_minor...\n");
    Matrix A = Matrix_zeros(3, 3);
    A.data[0][0] = 1; A.data[0][1] = 2; A.data[0][2] = 3;
    A.data[1][0] = 4; A.data[1][1] = 5; A.data[1][2] = 6;
    A.data[2][0] = 7; A.data[2][1] = 8; A.data[2][2] = 9;

    Matrix M = Matrix_minor(&A, 1, 1);
    assert(M.rows == 2 && M.cols == 2);
    assert(M.data[0][0] == 1.0);
    assert(M.data[0][1] == 3.0);
    assert(M.data[1][0] == 7.0);
    assert(M.data[1][1] == 9.0);

    Matrix_free(A);
    Matrix_free(M);
    printf("Matrix_minor test passed.\n");
}

void test_matrix_determinant() {
    printf("Testing Matrix_det...\n");
    Matrix A = Matrix_zeros(2, 2);
    A.data[0][0] = 1; A.data[0][1] = 2;
    A.data[1][0] = 3; A.data[1][1] = 4;

    double det = Matrix_det(&A);
    assert(fabs(det - (-2.0)) < 1e-9);

    Matrix_free(A);
    printf("Matrix_det test passed.\n");
}

void test_matrix_trace() {
    printf("Testing Matrix_trace...\n");
    Matrix A = Matrix_zeros(2, 2);
    A.data[0][0] = 1; A.data[0][1] = 2;
    A.data[1][0] = 3; A.data[1][1] = 4;

    double trace = Matrix_trace(&A);
    assert(fabs(trace - 5.0) < 1e-9);

    Matrix_free(A);
    printf("Matrix_trace test passed.\n");
}

void test_matrix_frobenius_norm() {
    printf("Testing Matrix_frobenius_norm...\n");
    Matrix A = Matrix_zeros(2, 2);
    A.data[0][0] = 1; A.data[0][1] = 2;
    A.data[1][0] = 3; A.data[1][1] = 4;

    double norm = Matrix_frobenius_norm(&A);
    assert(fabs(norm - sqrt(30.0)) < 1e-9);

    Matrix_free(A);
    printf("Matrix_frobenius_norm test passed.\n");
}

void test_matrix_free() {
    printf("Testing Matrix_free...\n");
    Matrix A = Matrix_zeros(2, 2);
    Matrix_free(A);
    printf("Matrix_free test passed.\n");
}

void test_matrix_inverse() {
    printf("Testing Matrix_inverse...\n");
    Matrix A = Matrix_zeros(3,3);
    A.data[0][0] = 1; A.data[0][1] = 2; A.data[0][2] = 3;
    A.data[1][0] = 0; A.data[1][1] = 2; A.data[1][2] = 0;
    A.data[2][0] = 0; A.data[2][1] = 0; A.data[2][2] = 3;

    Matrix I = Matrix_identity(3);
    Matrix AI = Matrix_inverse(&A);
    Matrix AAI = Matrix_multiply(&A, &AI);

    assert(matrix_equal(&AAI, &I));

    Matrix_free(A);
    Matrix_free(I);
    Matrix_free(AI);
    Matrix_free(AAI);
    printf("Matrix_inverse test passed\n");
}

void test_matrix_svd() {
    printf("Testing Matrix_svd...\n");
    Matrix A = Matrix_zeros(4, 3);
    A.data[0][0] = 1; A.data[0][1] = 2; A.data[0][2] = 3;
    A.data[1][0] = 0; A.data[1][1] = 2; A.data[1][2] = 0;
    A.data[2][0] = 0; A.data[2][1] = 0; A.data[2][2] = 3;
    A.data[3][0] = 8; A.data[3][1] = 0; A.data[3][2] = 3;

    SVDResult svd = Matrix_svd(&A);

    for (int i = 0; i < svd.Sigma.rows; ++i) {
        for (int j = 0; j < svd.Sigma.cols; ++j) {
            if (i != j) {
                assert(svd.Sigma.data[i][j] == 0 && "Sigma is not diagonal");
            }
        }
    }

    // Check if singular values are non-negative and sorted in non-increasing order
    for (int i = 1; i < svd.Sigma.rows; ++i) {
        assert(svd.Sigma.data[i][i] >= 0 && "Singular values should be non-negative");
        assert(svd.Sigma.data[i-1][i-1] >= svd.Sigma.data[i][i] && "Singular values should be sorted in non-increasing order");
    }

    // Reconstruct A from U, Sigma, V
    Matrix U = svd.U;
    Matrix Sigma = svd.Sigma;
    Matrix V = svd.V;
    Matrix VT = Matrix_transpose(&V);

    Matrix USigma = Matrix_multiply(&U, &Sigma);
    Matrix A_reconstructed = Matrix_multiply(&USigma, &VT);

    // Check if A is approximately equal to U Sigma V^T
    assert(matrix_equal(&A, &A_reconstructed));

    // Check orthogonality of U & V
    Matrix UT = Matrix_transpose(&U);
    Matrix UTU = Matrix_multiply(&UT, &U);
    Matrix I = Matrix_identity(UTU.rows);

    assert(matrix_equal(&I, &UTU));

    Matrix VTV = Matrix_multiply(&VT, &V);
    assert(matrix_equal(&I, &VTV));

    // Clean up
    Matrix_free(A);
    Matrix_free(USigma);
    Matrix_free(A_reconstructed);
    Matrix_free(U);
    Matrix_free(Sigma);
    Matrix_free(V);
    Matrix_free(VT);
    Matrix_free(UT);
    Matrix_free(UTU);
    Matrix_free(VTV);

    printf("Matrix_svd test passed!\n");
}

void test_matrix_solve() {
    printf("Testing Matrix_solve...\n");
    Matrix A = Matrix_zeros(3, 3);
    A.data[0][0] = 1.0; A.data[0][1] = 2.0; A.data[0][2] = 3.0;
    A.data[1][0] = 2.0; A.data[1][1] = 5.0; A.data[1][2] = 3.0;
    A.data[2][0] = 1.0; A.data[2][1] = 0.0; A.data[2][2] = 8.0;

    Matrix b = Matrix_zeros(3, 1);
    b.data[0][0] = 6.0;
    b.data[1][0] = 5.0;
    b.data[2][0] = 7.0;

    Matrix x = Matrix_solve(&A, &b);
    Matrix Ax = Matrix_multiply(&A, &x);

    assert(matrix_equal(&Ax, &b) && "Ax != b");


    // Free allocated matrices
    Matrix_free(A);
    Matrix_free(b);
    Matrix_free(x);
    Matrix_free(Ax);

    printf("Matrix_solve test passed!\n");
}

int main() {
    test_matrix_zeros();
    test_matrix_identity();
    test_matrix_scale();
    test_matrix_multiply();
    test_matrix_transpose();
    test_matrix_minor();
    test_matrix_determinant();
    test_matrix_trace();
    test_matrix_frobenius_norm();
    test_matrix_free();
    test_matrix_svd();
    test_matrix_inverse();
    test_matrix_solve();

    printf("All tests passed successfully.\n\n");
    return 0;
}
