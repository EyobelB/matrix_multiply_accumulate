#include <immintrin.h>
#include <stdbool.h>
#include <stdio.h>

// Compile with the -mfma option to ensure access to 256-bit registers
// Work under assumption of doube precision floating point numbers (float64)
// Must support AVX2 extension

// Define the matrix as a 1D vector to make matrix multiplication easier
typedef struct {
    __m256d* matrix_vals;
    int row_len;
    int column_len;
    int row_reg_count;
    int col_reg_count;
} matrix;

// Create a constructor for populating the matrix
matrix* create_matrix(int rows, int columns) {
    // Allocate space for the matrix pointer to be returned, define dimensions
    matrix* new_matrix = (matrix*)malloc(sizeof(matrix));
    new_matrix->row_len = rows;
    new_matrix->column_len = columns;

    // Allocate the necessary amount of 512 bit fused doubles
    new_matrix->row_reg_count = (rows % 4 == 0) ? (rows/4) : (rows/4)+1;
    new_matrix->col_reg_count = (columns % 4 == 0) ? (columns/4) : (columns/4)+1;
    new_matrix->matrix_vals = (__m256d*)malloc(sizeof(__m256d)*(new_matrix->row_reg_count+new_matrix->col_reg_count));

    // Fill these new rows and columns with zeros
    for (int reg = 0; reg < new_matrix->row_reg_count+new_matrix->col_reg_count; reg++) {
        new_matrix->matrix_vals[reg] = _mm256_setzero_pd();
    }

    // Return the created matrix
    printf("Matrix generated and initialized....\n");
    return new_matrix;
}

void delete_matrix(matrix* old_matrix) {
    // Free the matrix_vals part
    free(old_matrix->matrix_vals);
    old_matrix->matrix_vals = NULL;

    // Free the whole object
    free(old_matrix);
    old_matrix = NULL;
    printf("Matrix deleted....");
}

void set_matrix(matrix* matrix_to_change, double** new_vals) {
    // Assume the input is a 2D double array, this is a typical format
    //  for normal C matrices
    int matrix_indices = 0;
    for (int r = 0; r < matrix_to_change->row_len; r += 4) {
        for (int c = 0; c < matrix_to_change->column_len; c += 4) {
            // Create 1D array of doubles to add based on array size
            double insert_array[4] = {};

            // Fill the insert array
            for (int i = 0; i < 4; i++) {
                if (matrix_to_change->column_len - c <= i) {
                    insert_array[i] = 0.0;
                } else {
                    insert_array[i] = new_vals[r][i];
                }
            }

            // Add to the matrix array
            matrix_to_change->matrix_vals[matrix_indices] = _mm256_set_pd(insert_array[0], insert_array[1], insert_array[2], insert_array[3]);
            matrix_indices++;
        }
    }
    // Report back that matrix is set
    printf("Matrix set....\n");
}

matrix* matrix_multiply_accumulate(matrix* A, matrix* B, matrix* C) {
    // Determine if this is a valid MMA in the first place
    if (A->column_len != B->row_len) {
        printf("A's column count must match B's row count!\n");
        return NULL;
    }
    if (C->column_len != A->column_len || C->row_len != B->row_len) {
        printf("Ensure the accumulate matrix matches dimensions of multiplied matrix....\n");
        return NULL;
    }

    // Create a result matrix of the apporpriate size
    matrix* result = create_matrix(A->column_len, B->row_len);

    // Start computation with a triple nested for loop
    for (int r = 0; r < result->row_reg_count; r++) {
        for (int c = 0; c < result->col_reg_count; c++) {
            for (int i = 0; i < result->col_reg_count; i++) {
                result->matrix_vals[r*result->row_reg_count+i] = _mm256_fmadd_pd(A->matrix_vals[r*A->row_reg_count+c], B->matrix_vals[c*B->row_reg_count+i], C->matrix_vals[r*C->row_reg_count+i]);
            }
        }
    }

    // Return the result
    printf("MMA operation complete....\n");
    return result;
}

int main() {
    // Create three matrices
    matrix* A = create_matrix(8, 8);
	matrix* B = create_matrix(8, 8);
	matrix* C = create_matrix(8, 8);

    // Calculate A
	double** vals = (double**)malloc(sizeof(double*)*8);
	for (int i = 0; i < 8; i++) {
		vals[i] = (double*)malloc(sizeof(double)*8);
	}
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			vals[i][j] = 0.5*i*j;
		}
	}
	set_matrix(A, vals);

	// Calculate B
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			vals[i][j] = 0.3*i*j;
		}
	}
	set_matrix(B, vals);

	// Calculate C
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			vals[i][j] = 0.2*i*j;
		}
	}
    set_matrix(C, vals);

    // Calculate the answer and display it on the screen
    matrix* answer = matrix_multiply_accumulate(A, B, C);

    // Free variables
    delete_matrix(A);
    delete_matrix(B);
    delete_matrix(C);
    free(vals);
}