#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 100 // Define the matrix size
#define N 100 // Define the matrix size

// Define the number of threads to use.
int P = 10; // Replace 4 with the desired number of threads.

struct thread_args {
  int row;
  int column;
  int **matrix_a;
  int **matrix_b;
  int **matrix_c;
};

void *thread_function(void *args) {
  // Get the thread arguments.
  struct thread_args *thread_args = (struct thread_args *)args;

  // Calculate the element of the resulting matrix at the given row and column.
  thread_args->matrix_c[thread_args->row][thread_args->column] = 0;
  for (int i = 0; i < M; i++) {
    thread_args->matrix_c[thread_args->row][thread_args->column] +=
        thread_args->matrix_a[thread_args->row][i] *
        thread_args->matrix_b[i][thread_args->column];
  }

  // Return NULL to indicate that the thread has finished.
  return NULL;
}

int main() {
  // Create a timer variable.
  clock_t start_time, end_time;

  // Start the timer.
  start_time = clock();

  // Create the two input matrices.
  int **matrix_a = (int **)malloc(M * sizeof(int *));
  int **matrix_b = (int **)malloc(M * sizeof(int *));
  for (int i = 0; i < M; i++) {
    matrix_a[i] = (int *)malloc(N * sizeof(int));
    matrix_b[i] = (int *)malloc(N * sizeof(int));
  }

  // Initialize the input matrices with random values.
  srand(time(NULL)); // Seed the random number generator

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      matrix_a[i][j] = rand() % 100; // Generate a random number between 0 and 99
      matrix_b[i][j] = rand() % 100; // Generate a random number between 0 and 99
    }
  }

  // Create the resulting matrix.
  int **matrix_c = (int **)malloc(M * sizeof(int *));
  for (int i = 0; i < M; i++) {
    matrix_c[i] = (int *)malloc(N * sizeof(int));
  }

  // Create the thread arguments.
  struct thread_args *thread_args = (struct thread_args *)malloc(P *
                                                                  sizeof(struct thread_args));

  // Create the threads.
  pthread_t threads[P];
  for (int i = 0; i < P; i++) {
    thread_args[i].row = i;
    thread_args[i].matrix_a = matrix_a;
    thread_args[i].matrix_b = matrix_b;
    thread_args[i].matrix_c = matrix_c;

    pthread_create(&threads[i], NULL, thread_function, &thread_args[i]);
  }

  // Wait for all the threads to finish.
  for (int i = 0; i < P; i++) {
    pthread_join(threads[i], NULL);
  }

  // Stop the timer.
  end_time = clock();

  // Print the resulting matrix.
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%d ", matrix_c[i][j]);
    }
    printf("\n");
  }
  
  // Calculate the elapsed time and print it to the console.
  float elapsed_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %f seconds\n", elapsed_time);

  // Free the memory allocated for the matrices.
  for (int i = 0; i < M; i++) {
    free(matrix_a[i]);
    free(matrix_b[i]);
    free(matrix_c[i]);
  }
  free(matrix_a);
  free(matrix_b);
  free(matrix_c);

  // Free the memory allocated for the thread arguments.
  free(thread_args);

  return 0;
}
