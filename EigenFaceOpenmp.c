#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>
#include <omp.h>

#define IMAGE_WIDTH 92
#define IMAGE_HEIGHT 112
#define NUM_TRAINING_IMAGES 1
#define NUM_EIGENFACES 10

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

void calculateTranspose(double **matrixA, double **transposeA, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposeA[j][i] = matrixA[i][j];
        }
    }
}

double calculateDotProduct(double *vectorA, double *vectorB, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += vectorA[i] * vectorB[i];
    }
    return result;
}

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

void calculateCovariance(double **trainingImages, double **covarianceMatrix, double *meanFace, int numImages, int imageSize) {
    // Calculate mean face
    for (int i = 0; i < imageSize; ++i) {
        meanFace[i] = 0;
        for (int j = 0; j < numImages; ++j) {
            meanFace[i] += trainingImages[j][i];
        }
        meanFace[i] /= numImages;
    }

    // Subtract mean face from each image
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            trainingImages[i][j] -= meanFace[j];
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

void performEigenDecomposition(double **matrix, double **eigenvectors, int size, int numEigenvectors) {
    lapack_int matrixSize = size;
    lapack_int leadingDimensionA = size;
    double* eigenvalues = (double*)malloc(size * sizeof(double));

    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', matrixSize, *matrix, leadingDimensionA, eigenvalues);

    if (info > 0) {
        fprintf(stderr, "LAPACKE_dsyev failed with error %d\n", info);
        exit(EXIT_FAILURE);
    }

    // Extract the eigenvectors
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

int main() {
    omp_set_num_threads(4);

    double **trainingImages = (double **)malloc(NUM_TRAINING_IMAGES * sizeof(double *));
    for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
        trainingImages[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
    }
    readImages(trainingImages, "hand_images.txt");

    double **covarianceMatrix = (double **)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double *));
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        covarianceMatrix[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
    }

    double **eigenfaces = (double **)malloc(NUM_EIGENFACES * sizeof(double *));
    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        eigenfaces[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
    }

    double *meanFace = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));

    double startTime = getCurrentTimeInSeconds();

    calculateCovariance(trainingImages, covarianceMatrix, meanFace, NUM_TRAINING_IMAGES, IMAGE_WIDTH * IMAGE_HEIGHT);
    performEigenDecomposition(covarianceMatrix, eigenfaces, IMAGE_WIDTH * IMAGE_HEIGHT, NUM_EIGENFACES);

    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        normalizeVector(eigenfaces[i], IMAGE_WIDTH * IMAGE_HEIGHT);
    }

    double endTime = getCurrentTimeInSeconds();
    printf("Elapsed time: %.4f seconds\n", endTime - startTime);

    for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
        free(trainingImages[i]);
    }
    free(trainingImages);

    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        free(covarianceMatrix[i]);
    }
    free(covarianceMatrix);

    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        free(eigenfaces[i]);
    }
    free(eigenfaces);

    free(meanFace);

    return 0;
}
