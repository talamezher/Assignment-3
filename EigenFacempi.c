#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>
#include <omp.h>
#include <mpi.h>

#define IMAGE_WIDTH 64     // Width of the input images
#define IMAGE_HEIGHT 64   // Height of the input images
#define NUM_TRAINING_IMAGES 100   // Number of training images
#define NUM_EIGENHAND_AND_FINGERFACES 20   // Number of eigenhand and fingerfaces to use

// Function to read hand and finger image data from files
void readImages(double **trainingImages, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
        for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; ++j) {
            fscanf(file, "%lf", &trainingImages[i][j]);
        }
    }

    fclose(file);
}

// Function to calculate the transpose of a matrix
void calculateTranspose(double **matrixA, double **transposeA, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposeA[j][i] = matrixA[i][j];
        }
    }
}

// Function to calculate the dot product of two vectors
double calculateDotProduct(double *vectorA, double *vectorB, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += vectorA[i] * vectorB[i];
    }
    return result;
}

// Function to normalize a vector
void normalizeVector(double *vector, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += vector[i] * vector[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < size; ++i) {
        vector[i] /= norm;
    }
}

// Function to calculate the covariance matrix
void calculateCovariance(double **trainingImages, double **covarianceMatrix, double *meanHandAndFingerFace, int numImages, int imageSize) {
    // Calculate mean hand and finger face
    for (int i = 0; i < imageSize; ++i) {
        meanHandAndFingerFace[i] = 0;
        for (int j = 0; j < numImages; ++j) {
            meanHandAndFingerFace[i] += trainingImages[j][i];
        }
        meanHandAndFingerFace[i] /= numImages;
    }

    // Subtract mean hand and finger face from each image
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            trainingImages[i][j] -= meanHandAndFingerFace[j];
        }
    }

    // Calculate covariance matrix
    for (int i = 0; i < imageSize; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            covarianceMatrix[i][j] = 0;
            for (int k = 0; k < numImages; ++k) {
                covarianceMatrix[i][j] += trainingImages[k][i] * trainingImages[k][j];
            }
            covarianceMatrix[i][j] /= (numImages - 1);
        }
    }
}

// Function to perform eigenvalue decomposition using LAPACKE
void performEigenDecomposition(double **matrix, double **eigenvectors, int size, int numEigenvectors) {
    // LAPACKE variables
    lapack_int matrixSize = size;
    lapack_int leadingDimensionA = size;
    double* eigenvalues = (double*)malloc(size * sizeof(double));

    // LAPACKE_dsyev function for eigenvalue decomposition
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', matrixSize, *matrix, leadingDimensionA, eigenvalues);

    // Check for errors in LAPACKE_dsyev
    if (info > 0) {
        fprintf(stderr, "LAPACKE_dsyev failed with error %d\n", info);
        exit(EXIT_FAILURE);
    }

    // Extract the eigenhand and fingerfaces
    for (int i = 0; i < numEigenvectors; ++i) {
        for (int j = 0; j < size; ++j) {
            eigenvectors[i][j] = matrix[j][i];
        }
    }

    free(eigenvalues);
}

double getCurrentTimeInSeconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Read training images from file
        double **trainingImages = (double **)malloc(NUM_TRAINING_IMAGES * sizeof(double *));
        for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
            trainingImages[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
        }
        readImages(trainingImages, "hand_images.txt");

        // Create space for covariance matrix, eigenhand and fingerfaces, and mean face
        double **covarianceMatrix = (double **)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double *));
        for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
            covarianceMatrix[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
        }

        double **eigenhandAndFingerfaces = (double **)malloc(NUM_EIGENHAND_AND_FINGERFACES * sizeof(double *));
        for (int i = 0; i < NUM_EIGENHAND_AND_FINGERFACES; ++i) {
            eigenhandAndFingerfaces[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
        }

        double *meanHandAndFingerFace = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));

        // Calculate covariance matrix
        calculateCovariance(trainingImages, covarianceMatrix, meanHandAndFingerFace, NUM_TRAINING_IMAGES, IMAGE_WIDTH * IMAGE_HEIGHT);

        // Perform eigenvalue decomposition to get eigenhand and fingerfaces
        performEigenDecomposition(covarianceMatrix, eigenhandAndFingerfaces, IMAGE_WIDTH * IMAGE_HEIGHT, NUM_EIGENHAND_AND_FINGERFACES);

        // Normalize the eigenhand and fingerfaces
        for (int i = 0; i < NUM_EIGENHAND_AND_FINGERFACES; ++i) {
            normalizeVector(eigenhandAndFingerfaces[i], IMAGE_WIDTH * IMAGE_HEIGHT);
        }

        // Free allocated memory
        for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
            free(trainingImages[i]);
        }
        free(trainingImages);

        for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
            free(covarianceMatrix[i]);
        }
        free(covarianceMatrix);

        for (int i = 0; i < NUM_EIGENHAND_AND_FINGERFACES; ++i) {
            free(eigenhandAndFingerfaces[i]);
        }
        free(eigenhandAndFingerfaces);

        free(meanHandAndFingerFace);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
