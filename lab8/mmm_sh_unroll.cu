/*

nvcc  mmm_sh_unroll.cu -o mmm_sh_unroll

*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

// set the matrix_size, block_num and the thread
const int MATRIX_SIZE = 2048;
#define TILE_WIDTH 16

// const int NUM_BLOCKS = 4096;

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans)                           \
    {                                                 \
        gpuAssert((ans), (char *)__FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void kernel_mmm(float *arr1, float *arr2, float *arr3, int arrlen)
{
    __shared__ float Mds0[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds0[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Mds1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds1[TILE_WIDTH][TILE_WIDTH];
  
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue0 = 0; // register
    float Pvalue1 = 0; // register
    int m = 0;

    for (; m < arrlen / TILE_WIDTH; m+=2)
    {
      Mds0[ty][tx] = arr1[Row * arrlen + (m * TILE_WIDTH + tx)];
      Nds0[ty][tx] = arr2[Col + (m * TILE_WIDTH + ty) * arrlen];
      Mds1[ty][tx] = arr1[Row * arrlen + ((m+1) * TILE_WIDTH + tx)];
      Nds1[ty][tx] = arr2[Col + ((m+1) * TILE_WIDTH + ty) * arrlen];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; k++)
      {
          Pvalue0 += Mds0[ty][k] * Nds0[k][tx];
          Pvalue1 += Mds1[ty][k] * Nds1[k][tx];
      }
      __syncthreads();
    }


    for (; m < arrlen / TILE_WIDTH; m++)
    {
      // calculate the remaining elements 
      Mds0[ty][tx] = arr1[Row * arrlen + (m * TILE_WIDTH + tx)];
      Nds0[ty][tx] = arr2[Col + (m * TILE_WIDTH + ty) * arrlen];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; k++){
          Pvalue0 += Mds0[ty][k] * Nds0[k][tx];
      }
      __syncthreads();
    
    }

    arr3[Row * arrlen + Col] = Pvalue0 + Pvalue1;
}

void initialize_Array_1D(float *arr, int len, int seed);
void MMM_CPU(float *arr1, float *arr2, float *arr3, int len);
void write_2d_Array_To_File(float **arr, int x_len, int y_len, char *filename);


void initialize_Array_1D(float *arr, int len, int seed)
{
    int i, j;
    float randNum;
    srand(seed);

    for (i = 0; i < len; i++)
    {
        for (j = 0; j < len; j++)
        {
            randNum = (float)rand();
            arr[i * MATRIX_SIZE + j] = randNum;
        }
    }
}

void MMM_CPU(float *arr1, float *arr2, float *arr3, int len)
{
    int i, j, k = 0;
    float sum;

    for (i = 0; i < len; i++)
    {
        for (j = 0; j < len; j++)
        {
            sum = 0;
            for (k = 0; k < len; k++)
            {
                sum += arr1[i * len + k] * arr2[k * len + j];
            }
            arr3[i * len + j] += sum;
        }
    }
}

double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) * 1.0e3 + ((double)temp.tv_nsec) * 1.0e-6);
}

void write_2d_Array_To_File(float *arr, int x_len, int y_len, char *filename)
{
    int i, j;
    FILE *f = fopen(filename, "w");
    for (i = 0; i < x_len; i++)
    {
        for (j = 0; j < y_len - 1; j++)
        {
            fprintf(f, "%f, ", arr[i * MATRIX_SIZE + j]);
        }
        fprintf(f, "%f\n", arr[i * MATRIX_SIZE + j - 1]);
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    // loop variables
    int i, j, errors = 0;

    float maxerror = 0;

    // timing variables
    cudaEvent_t start1, stop1;
    float elapsed_gpu1;
    cudaEvent_t start2, stop2;
    float elapsed_gpu2;
    struct timespec time_start, time_stop;
    double elapse_time;

    // Arrays on GPU global memoryc
    float *g_x;
    float *g_y;
    float *g_result;

    // Arrays on the host memory
    float *h_x;
    float *h_y;
    float *h_result_GPU;
    float *h_result_CPU;

    FILE *f = fopen("mismatches.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    size_t allocSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    printf("init all done\n");

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // Allocate arrays on GPU memory
    CUDA_SAFE_CALL(cudaMalloc((void **)&g_x, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&g_y, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&g_result, allocSize));

    // Allocate arrays on host memory
    h_x = (float *)malloc(allocSize);
    h_y = (float *)malloc(allocSize);
    h_result_GPU = (float *)malloc(allocSize);
    h_result_CPU = (float *)malloc(allocSize);

    printf("\nInitializing the arrays ...");

    // initialize host arrays
    initialize_Array_1D(h_x, MATRIX_SIZE, 2453);
    initialize_Array_1D(h_y, MATRIX_SIZE, 1467);


    // Create the cuda events
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    // Record event on the default stream
    cudaEventRecord(start1, 0);

    CUDA_SAFE_CALL(cudaMemcpy(g_x, h_x, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(g_y, h_y, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(g_result, h_result_GPU, allocSize, cudaMemcpyHostToDevice));


    // 1024/16 = 64
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(128, 128, 1);


    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);


    kernel_mmm<<<dimGrid, dimBlock>>>(g_x, g_y, g_result, MATRIX_SIZE);

    // record the time
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    // print the time
    cudaEventElapsedTime(&elapsed_gpu2, start2, stop2);
    printf("\nGPU time for just the MMM (kernel) execution: %f (msec)\n", elapsed_gpu2);
    // destroy
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    CUDA_SAFE_CALL(cudaPeekAtLastError());


    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_result_GPU, g_result, allocSize, cudaMemcpyDeviceToHost));

    // record the time
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    // print the time
    cudaEventElapsedTime(&elapsed_gpu1, start1, stop1);
    printf("\nGPU time including data transfers: %f (msec)\n", elapsed_gpu1);
    // destroy
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    printf("GPU calculation ended\n");

    // CPU calculation time

    clock_gettime(CLOCK_REALTIME, &time_start);
    MMM_CPU(h_x, h_y, h_result_CPU, MATRIX_SIZE);
    clock_gettime(CLOCK_REALTIME, &time_stop);

    elapse_time = interval(time_start, time_stop);
    printf("\nCPU time: %f(msec)\n", (float)(elapse_time));
    printf("CPU calculation ended\n");

    write_2d_Array_To_File(h_result_GPU, MATRIX_SIZE, MATRIX_SIZE, "GPU_output.txt");
    write_2d_Array_To_File(h_result_CPU, MATRIX_SIZE, MATRIX_SIZE, "CPU_output.txt");

    for (i = 0; i < MATRIX_SIZE; i++)
    {
        for (j = 0; j < MATRIX_SIZE; j++)
        {
            int index = i * MATRIX_SIZE + j;
            if (h_result_CPU[index] != h_result_GPU[index])
            {
                fprintf(f, "Mismatch at [%d,%d] GPU = %f CPU = %f at a difference of %f\n", i, j, h_result_GPU[i * MATRIX_SIZE + j], h_result_CPU[i * MATRIX_SIZE + j], h_result_GPU[i * MATRIX_SIZE + j] - h_result_CPU[i * MATRIX_SIZE + j]);
                if(h_result_GPU[i * MATRIX_SIZE + j] - h_result_CPU[i * MATRIX_SIZE + j] > maxerror){
                    maxerror = h_result_GPU[i * MATRIX_SIZE + j] - h_result_CPU[i * MATRIX_SIZE + j];
                }

            }
            if(h_result_CPU[index] >= h_result_GPU[index]*1.02 or  h_result_CPU[index] <= h_result_GPU[index]*0.98){
                errors++;
            }
        }
    }

    printf("results check finished\n");

    printf("Found %d differences in a range of 1%\n", errors);
    printf("The max error is %f\n", maxerror);


    CUDA_SAFE_CALL(cudaFree(g_x));
    CUDA_SAFE_CALL(cudaFree(g_y));
    CUDA_SAFE_CALL(cudaFree(g_result));

    free(h_x);
    free(h_y);
    free(h_result_GPU);
    free(h_result_CPU);

    return 0;
}