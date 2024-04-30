#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <lapacke.h>

// Other function declarations and constants remain unchanged

void calculateCovariance(double **trainingImages, double **covarianceMatrix, double *meanFace, int numImages, int imageSize) {
    // Calculate mean face in parallel
    #pragma omp parallel for
    for (int i = 0; i < imageSize; ++i) {
        meanFace[i] = 0;
        for (int j = 0; j < numImages; ++j) {
            meanFace[i] += trainingImages[j][i];
        }
        meanFace[i] /= numImages;
    }

    // Subtract mean face from each image in parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            trainingImages[i][j] -= meanFace[j];
        }
    }

    // Calculate covariance matrix in parallel
    #pragma omp parallel for collapse(2)
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

int main() {
    // Other parts of the main function remain unchanged

    double startTime = getCurrentTimeInSeconds();

    // Parallelize the computation of covariance matrix and mean face
    calculateCovariance(trainingImages, covarianceMatrix, meanFace, NUM_TRAINING_IMAGES, IMAGE_WIDTH * IMAGE_HEIGHT);
    
    // Perform eigenvalue decomposition (this part may not directly support OpenMP parallelism)
    performEigenDecomposition(covarianceMatrix, eigenfaces, IMAGE_WIDTH * IMAGE_HEIGHT, NUM_EIGENFACES);

    // Normalize eigenfaces in parallel
    #pragma omp parallel for
    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        normalizeVector(eigenfaces[i], IMAGE_WIDTH * IMAGE_HEIGHT);
    }

    double endTime = getCurrentTimeInSeconds();
    printf("Elapsed time: %.4f seconds\n", endTime - startTime);

    // Free memory and other cleanup remain unchanged

    return 0;
}

