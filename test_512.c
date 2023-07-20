#include <immintrin.h>
#include <stdbool.h>
#include <stdio.h>

// Make sure you compile with the -mavx512f option for compatibility with both AMD and Intel processors
// Work under assumption of double precision floating point numbers (float64, AKA doubles)
// Must support AVX-512 extension

// Define the matrix structure using intrinsic registers
typedef struct {
    __m512d** two_dim_matrix_vals;
    int row_len;
    int column_len;
    int remainder;
} matrix;

// Create a constructor for the matrix
matrix* create_matrix(int rows, int columns) {
    // Allocate space for the matrix pointer to be returned, define r/c size
    matrix* new_matrix = (matrix*)malloc(sizeof(matrix));
    new_matrix->row_len = rows;
    new_matrix->column_len = columns;

    // Check if we need another remainder register or not
    int remainder_reg_count = 0;
    if (columns % 8 != 0) {
        remainder_reg_count = 1;
    }
    new_matrix->remainder = remainder_reg_count;

    // Allocate the amount of necessary matrix values
    // Basically, do row and (column/8 + 1) or column/8 if no remainder
    new_matrix->two_dim_matrix_vals = (__m512d**)malloc(sizeof(__m512d*)*rows);
    for (int r = 0; r < rows; r++) {
        // Allocate the columns
        new_matrix->two_dim_matrix_vals[r] = (__m512d*)malloc(sizeof(__m512d)*((columns/8) + remainder_reg_count));
        
        // Initialize to zero....
        for (int c = 0; c < (columns/8) + remainder_reg_count; c++) {
            new_matrix->two_dim_matrix_vals[r][c] = _mm512_setzero_pd();
        }
    }

    // Matrix generation complete
    printf("Matrix generated and initialized....\n");
    return new_matrix;
}

void delete_matrix(matrix* old_matrix) {
    // Free the rows
    for (int r = 0; r < old_matrix->row_len; r++) {
        free(old_matrix->two_dim_matrix_vals[r]);
        old_matrix->two_dim_matrix_vals[r] = NULL;
    }

    // Free the whole pointer
    free(old_matrix->two_dim_matrix_vals);
    old_matrix->two_dim_matrix_vals = NULL;

    // Free the entire matrix object
    free(old_matrix);
    old_matrix = NULL;
    printf("Matrix deleted....\n");
}

void set_matrix(matrix* matrix_to_change, double** new_vals) {
    // Create double for loop to set registers
    for (int r = 0; r < matrix_to_change->row_len; r++) {
        for (int c = 0; c < matrix_to_change->column_len/8 + matrix_to_change->remainder; c += 8) {
            // Create 1D array of doubles to add based on array size
            double insert_array[8] = {};

            // Fill the insert array
            for (int i = 0; i < 8; i++) {
                if (matrix_to_change->column_len < i) {
                    insert_array[i] = 0.0;
                } else {
                    insert_array[i] = new_vals[r][i];
                }
            }

            // Use set_pd function
            matrix_to_change->two_dim_matrix_vals[r][c] = _mm512_set_pd(insert_array[0], insert_array[1], insert_array[2], insert_array[3], insert_array[4], insert_array[5], insert_array[6], insert_array[7]);
        }
    }
    // Report back that the matrix is set
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

    // Create a result matrix of the appropriate size
    matrix* result = create_matrix(A->column_len, B->row_len);

    // Start computation
    for (int r = 0; r < result->row_len; r++) {
		for (int c = 0; c < result->column_len/8 + result->remainder; c++) {
			result->two_dim_matrix_vals[r][c] = _mm512_fmadd_pd(A->two_dim_matrix_vals[r][c], B->two_dim_matrix_vals[r][c], C->two_dim_matrix_vals[r][c]);
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
    for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("For r=%d c=%d: %lf\n", i, j, answer->two_dim_matrix_vals[i][j]);
		}
	}

    // Free variables
    delete_matrix(A);
    delete_matrix(B);
    delete_matrix(C);
    free(vals);
}