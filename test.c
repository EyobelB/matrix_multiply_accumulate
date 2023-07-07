#include <immintrin.h>
#include <stdbool.h>
#include <stdio.h>


// Make sure you compile with the -mfma option for compatibility with both AMD and Intel processors

typedef struct {
	__m128d** two_dim_matrix_vals;
	int row_len;
	int column_len;
} matrix;

matrix* create_matrix(int rows, int columns) {
	matrix* new_matrix = (matrix*)malloc(sizeof(matrix));
	new_matrix->row_len = rows;
	new_matrix->column_len = columns;
	
	new_matrix->two_dim_matrix_vals = (__m128d**)malloc(sizeof(__m128d*)*rows);
	for (int r = 0; r < rows; r++) {
		new_matrix->two_dim_matrix_vals[r] = (__m128d*)malloc(sizeof(__m128d)*columns);
		for (int c = 0; c < columns; c++) {
			new_matrix->two_dim_matrix_vals[r][c] = _mm_setzero_pd();
		}
	}
	printf("Matrix generated....\n");
	return new_matrix;
}

void delete_matrix(matrix* old_matrix) {
	for (int r = 0; r < old_matrix->row_len; r++) {
		free(old_matrix->two_dim_matrix_vals[r]);
		old_matrix->two_dim_matrix_vals[r] = NULL;
	}
	free(old_matrix->two_dim_matrix_vals);
	old_matrix->two_dim_matrix_vals = NULL;
	free(old_matrix);
	old_matrix = NULL;
	printf("Matrix deleted....\n");
}

void set_matrix(matrix* old_matrix, double** new_vals) {
	for (int r = 0; r < old_matrix->row_len; r++) {
		for (int c = 0; c < old_matrix->column_len; c++) {
			// Problem on new_vals call
			old_matrix->two_dim_matrix_vals[r][c] = _mm_set1_pd(new_vals[r][c]);
		}
	}
	printf("Matrix set....\n");
}

matrix* matrix_multiply_accumulate(matrix* A, matrix* B, matrix* C) {
	if (A->column_len != B->row_len) {
		printf("Incorrect size of matrices to be multiplied, ensure that column count of first matrix matches row count of second....\n");
		return NULL;
	}
	matrix* result = create_matrix(A->row_len, B->column_len);

	if (C->column_len != result->column_len || C->row_len != result->row_len) {
		printf("Incorrect size of accumulate matrix, ensure that the row length matches the second matrix and the column length matches the first....\n");
		return NULL;
	}

	for (int r = 0; r < result->row_len; r++) {
		for (int c = 0; c < result->column_len; c++) {
			result->two_dim_matrix_vals[r][c] = _mm_fmadd_pd(A->two_dim_matrix_vals[r][c], B->two_dim_matrix_vals[r][c], C->two_dim_matrix_vals[r][c]);
		}
	}
	printf("MMA operation complete....\n");
	return result;
}

__m128d* fill_vector(double* vector_vals, int val_num) {
	__m128d* vector = malloc(sizeof(__m128d) * val_num);
	for (int i = 0; i < val_num; i++) {
		vector[i] = _mm_set1_pd(vector_vals[i]);
	}
	printf("Vector filled....\n");
	return vector;
}

int main() {
	matrix* A = create_matrix(3, 3);
	matrix* B = create_matrix(3, 3);
	matrix* C = create_matrix(3, 3);

	// Calculate A
	double** vals = (double**)malloc(sizeof(double*)*3);
	for (int i = 0; i < 3; i++) {
		vals[i] = (double*)malloc(sizeof(double)*3);
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			vals[i][j] = 0.5*i*j;
		}
	}
	set_matrix(A, vals);

	// Calculate B
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			vals[i][j] = 0.3*i*j;
		}
	}
	set_matrix(B, vals);

	// Calculate C
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			vals[i][j] = 0.2*i*j;
		}
	}
	set_matrix(C, vals);

	matrix* answer = matrix_multiply_accumulate(A, B, C);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("For r=%d c=%d: %lf\n", i, j, answer->two_dim_matrix_vals[i][j]);
		}
	}

	delete_matrix(A);
	delete_matrix(B);
	free(C);
	free(vals);
}