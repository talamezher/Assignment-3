#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 100 // Define the matrix size
#define N 100 // Define the matrix size

int main() {
  // Create a timer variable
  clock_t start_time, end_time;

  // Start the timer
  start_time = clock();

  // Create the two input matrices
  int **matrix_a = (int **)malloc(M * sizeof(int *));
  int **matrix_b = (int **)malloc(M * sizeof(int *));
  for (int i = 0; i < M; i++) {
    matrix_a[i] = (int *)malloc(N * sizeof(int));
    matrix_b[i] = (int *)malloc(N * sizeof(int));
  }

  // Initialize the input matrices with random values
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      matrix_a[i][j] = rand();
      matrix_b[i][j] = rand();
    }
  }

  // Create the resulting matrix
  int **matrix_c = (int **)malloc(M * sizeof(int *));
  for (int i = 0; i < M; i++) {
    matrix_c[i] = (int *)malloc(N * sizeof(int));
  }

  // Calculate the resulting matrix sequentially
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      matrix_c[i][j] = 0;
      for (int k = 0; k < M; k++) {
        matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
      }
    }
  }

  // Stop the timer
  end_time = clock();

  // Print the resulting matrix
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%d ", matrix_c[i][j]);
    }
    printf("\n");
  }
  
  // Calculate the elapsed time and print it to the console
  float elapsed_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %f seconds\n", elapsed_time);

  // Free the memory allocated for the matrices
  for (int i = 0; i < M; i++) {
    free(matrix_a[i]);
    free(matrix_b[i]);
    free(matrix_c[i]);
  }
  free(matrix_a);
  free(matrix_b);
  free(matrix_c);

  return 0;
}
