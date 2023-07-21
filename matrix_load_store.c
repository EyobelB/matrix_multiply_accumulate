#include <immintrin.h>
#include <stdbool.h>
#include <stdio.h>

#define DOUBLES_PER_REGISTER 4

// Compile with the -mfma option to ensure access to 256-bit registers
// Work under assumption of doube precision floating point numbers (float64)
// Must support AVX2 extension

// Define the matrix as a 2D vector with accessible row and column dimensions
typedef struct {
    double** matrix_vals;
    int row_len;
    int column_len;
} matrix;

// Create a matrix constructor
matrix* create_matrix(int row_count, int column_count) {
    // Allocate space for the object and rows first
    matrix* new_matrix = (matrix*)malloc(sizeof(matrix));
    new_matrix->matrix_vals = (double**)malloc(sizeof(double*)*row_count);

    // Set row and column lengths
    new_matrix->row_len = row_count;
    new_matrix->column_len = column_count;

    // Allocate each column's array
    for (int i = 0; i < column_count; i++) {
        new_matrix->matrix_vals[i] = (double*)malloc(sizeof(double)*column_count);
    }

    // Return created matrix
    printf("Matrix generated....\n");
    return new_matrix;
}

// Create matrix destructor
void delete_matrix(matrix* old_matrix) {
    // Free each column in the matrix, replace with NULL
    for (int i = 0; i < old_matrix->column_len; i++) {
        free(old_matrix->matrix_vals[i]);
        old_matrix->matrix_vals[i] = NULL;
    }

    // Free the 2D pointer for matrix values and the object itself
    free(old_matrix->matrix_vals);
    old_matrix->matrix_vals = NULL;
    free(old_matrix);
    old_matrix = NULL;

    // Indicate deletion is completed
    printf("Matrix deleted....\n");
}

// Implement accelerate matrix setting using AVX intrinsics
void set_matrix(matrix* old_matrix, double** new_vals) {
    // For each row, load in 4 column values at a time if possible
    for (int i = 0; i < old_matrix->row_len; i++) {
        // Then store all 4 back to the old_matrix array
        for (int j = 0; j < old_matrix->column_len; j += DOUBLES_PER_REGISTER) {
            // If we can fit 4 values, do so
            if ((old_matrix->column_len - j) / DOUBLES_PER_REGISTER >= 1) {
                // Define the doubles to be stored as an intermediate vector
                printf("%x", &new_vals[i][j]);
                __m256d doubles_to_store = _mm256_load_pd(&new_vals[i][j]);
                _mm256_store_pd(old_matrix->matrix_vals[j+(i*DOUBLES_PER_REGISTER)], doubles_to_store);
            } else {
                // If not, store the remaining by simply copying
                for (int k = j; k < old_matrix->column_len; k++) {
                    old_matrix->matrix_vals[i][k] = new_vals[i][k];
                }
            }
        }
    }
    // Indicate setting is complete
    printf("Matrix set....\n");
}



int main() {
    // Initialize values
    matrix* A = create_matrix(8, 8);
    double** vals_to_set = (double**)malloc(sizeof(double*)*8);
    for (int q = 0; q < 8; q++) {
        vals_to_set[q] = (double*)malloc(sizeof(double)*8);
        for (int w = 0; w < 8; w++) {
            vals_to_set[q][w] = w*q;
        }
    }


    // Set these values in the matrix
    set_matrix(A, vals_to_set);

    // Print matrix values
    for (int i = 0; i < A->row_len; i++) {
        for (int j = 0; j < A->column_len; j++) {
            printf("%lf ", A->matrix_vals[i][j]);
        }
        printf("\n");
    }

    // Free the matrix
    delete_matrix(A);
}