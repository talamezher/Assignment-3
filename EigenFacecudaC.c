#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define IMAGE_WIDTH 175
#define IMAGE_HEIGHT 200
#define NUM_IMAGES 1
#define NUM_EIGENFACES 10
#define BLOCK_SIZE 8

__global__ void calculateCovarianceMatrixKernel(double *deviceImages, double *deviceCovarianceMatrix, double *deviceMeanFace, int numImages, int imageSize) {
    int columnIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (columnIndex < imageSize) {
        // Calculate mean face
        deviceMeanFace[columnIndex] = 0;
        for (int rowIndex = 0; rowIndex < numImages; ++rowIndex) {
            deviceMeanFace[columnIndex] += deviceImages[rowIndex * imageSize + columnIndex];
        }
        deviceMeanFace[columnIndex] /= numImages;

        // Subtract mean face from each image
        for (int rowIndex = 0; rowIndex < numImages; ++rowIndex) {
            deviceImages[rowIndex * imageSize + columnIndex] -= deviceMeanFace[columnIndex];
        }
        // Calculate covariance matrix (omitted for elapsed time measurement)
    }
}

void readImages(double *images, const char *filename, int numImages, int imageSize) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int imageIndex = 0; imageIndex < numImages; ++imageIndex) {
        for (int pixelIndex = 0; pixelIndex < imageSize; ++pixelIndex) {
            fscanf(file, "%lf", &images[imageIndex * imageSize + pixelIndex]);
        }
    }

    fclose(file);
}

int main() {
    int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;
    double *hostImages = (double *)malloc(NUM_IMAGES * imageSize * sizeof(double));
    readImages(hostImages, "hand_images.txt", NUM_IMAGES, imageSize);

    double *deviceImages, *deviceCovarianceMatrix, *deviceMeanFace;
    cudaMalloc((void **)&deviceImages, NUM_IMAGES * imageSize * sizeof(double));
    cudaMalloc((void **)&deviceCovarianceMatrix, imageSize * imageSize * sizeof(double));
    cudaMalloc((void **)&deviceMeanFace, imageSize * sizeof(double));

    cudaMemcpy(deviceImages, hostImages, NUM_IMAGES * imageSize * sizeof(double), cudaMemcpyHostToDevice);
    double *hostMeanFace = (double *)malloc(imageSize * sizeof(double));

    clock_t start = clock();

    dim3 blocks((imageSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    calculateCovarianceMatrixKernel<<<blocks, threads>>>(deviceImages, deviceCovarianceMatrix, deviceMeanFace, NUM_IMAGES, imageSize);
    cudaDeviceSynchronize();

    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    free(hostImages);
    free(hostMeanFace);
    cudaFree(deviceImages);
    cudaFree(deviceCovarianceMatrix);
    cudaFree(deviceMeanFace);

    return 0;
}

