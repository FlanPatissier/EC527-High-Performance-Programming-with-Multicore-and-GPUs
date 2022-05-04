/*
    Steps:
    1. initialize a 2K⨉2K array
    2. Transfer the array to the GPU
    3. Create a single block of 16x16 threads
    4. Run 2000 iterations of SOR
    5. Transfer the output back to the CPU
    6. Do the timing
    7. Compare the results

*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

// set the matrix_size, block_num and the thread
const int MATRIX_SIZE = 2000;
const int NUM_BLOCKS = 1;
const int THREADS_PER_BLOCK_X = 16;
const int THREADS_PER_BLOCK_Y = 16;

// set the OMEGA and the iteration number for the SOR
const int SOR_ITERATIONS = 2000;
const int OMEGA = 1;

// for the GPU init and transfer
#define DEBUG_PRINT
#define ALLOCATE_AND_INIT
#define GPU_TIMING
#define TRANSFER_TO_GPU
#define LAUNCH_KERNEL
#define TRANSFER_RESULTS
#define FREE_MEMORY

// for the CPU init and transfer
#define COMPUTE_CPU_RESULTS
#define CPU_TIMING

// write and compare the results
#define COMPARE_RESULTS
#define WRITE_2D_ARRAYS

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

void initialize_Array_2D(float **arr, int len, int seed);
void write_2d_Array_To_File(float **arr, int x_len, int y_len, char *filename);
void SOR_CPU(float **arr, int OMEGA, int x_len, int y_len);

// __global__ void SOR_GPU(float *arr, int OMEGA, int x_len, int y_len)
// {
//     int i, j = 0;
//     float change;

//     int local_tid_x = threadIdx.x;
//     int local_tid_y = threadIdx.y;

//     int block_row = blockDim.x;
//     int block_col = blockDim.y;

//     int MATRIX_SIZE = 2000;

//     for (int k = 0; k < SOR_ITERATIONS; k++)
//     {
//         for (i = local_tid_x; i < MATRIX_SIZE; i += block_row)
//         {
//             for (j = local_tid_y; j < MATRIX_SIZE; j += block_col)
//             {
//                 if (i > 0 && i < MATRIX_SIZE - 1 && j > 0 && j < MATRIX_SIZE - 1)
//                 {
//                     change = arr[i * MATRIX_SIZE + j] - 0.25 * (arr[(i - 1) * MATRIX_SIZE + j] + arr[(i + 1) * MATRIX_SIZE + j] + arr[i * MATRIX_SIZE + (j - 1)] + arr[i * MATRIX_SIZE + (j + 1)]);
//                     // __syncthreads();
//                     arr[i * MATRIX_SIZE + j] -= (change * OMEGA);
//                     // __syncthreads();
//                 }
//             }
//         }
//     }
// }

__global__ void SOR_GPU(float *arr, int OMEGA, int x_len, int y_len)
{
    int i, j = 0;
    float change;
    int ii, jj = 0;

    int local_tid_x = threadIdx.x;
    int local_tid_y = threadIdx.y;

    int block_row = blockDim.x;
    int block_col = blockDim.y;

    int MATRIX_SIZE = 2000;


    for (int k = 0; k < SOR_ITERATIONS; k++)
    {
        if((local_tid_x == 0 && local_tid_y == 0) or (local_tid_x == 0 && local_tid_y == 8) or (local_tid_x == 8 && local_tid_y == 0) or (local_tid_x == 8 && local_tid_y == 8))
        {
            for(int i = local_tid_x; i < MATRIX_SIZE; i += block_col)
            {
                for(int ii = i; ii < i + 8; ii++)
                {
                    for( int j = local_tid_y; j < MATRIX_SIZE; j += block_row)
                    {
                        for(int jj = j; jj < j + 8; jj++)
                        {
                            if (ii > 0 && ii < MATRIX_SIZE - 1 && jj > 0 && jj < MATRIX_SIZE - 1)
                            {
                                change = arr[ii * MATRIX_SIZE + jj] - 0.25 * (arr[(ii - 1) * MATRIX_SIZE + jj] + arr[(ii + 1) * MATRIX_SIZE + jj] + arr[ii * MATRIX_SIZE + (jj - 1)] + arr[ii * MATRIX_SIZE + (jj + 1)]);
                                arr[ii * MATRIX_SIZE + jj] -= (change * OMEGA);
                            }

                        }
                    }

                }
            }
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

/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_REALTIME, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_REALTIME, &time_stop);
      measurement = interval(time_start, time_stop);

 */

void SOR_CPU(float *arr, int OMEGA, int x_len, int y_len)
{
    float change;
    int MATRIX_SIZE = 2000;

    for (int i = 0; i < x_len; i++)
    {
        for (int j = 0; j < y_len; j++)
        {
            if (i > 0 && i < x_len - 1 && j > 0 && j < y_len - 1)
            {
                change = arr[i * MATRIX_SIZE + j] - 0.25 * (arr[(i - 1) * MATRIX_SIZE + j] + arr[(i + 1) * MATRIX_SIZE + j] + arr[i * MATRIX_SIZE + (j - 1)] + arr[i * MATRIX_SIZE + (j + 1)]);
                arr[i * MATRIX_SIZE + j] -= change * OMEGA;
            }
        }
    }
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

void initialize_Array_2D(float *arr, int len, int seed)
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

// main function
int main(int argc, char **argv)
{
    // loop variables
    int i, j, errors = 0;
    int MATRIX_SIZE = 2000;

    // timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;
    struct timespec time_start, time_stop;
    double elapse_time;

    dim3 dimGrid(NUM_BLOCKS, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);


    // Arrays on GPU global memory
    float *g_A;

    // Arrays on host memory
    float *h_A_GPU;
    float *h_A_CPU;

    FILE *f = fopen("mismatches.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    size_t allocSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

#ifdef DEBUG_PRINT
    printf("init all done\n");
#endif

// Allocate arrays on GPU and host memory
#ifdef ALLOCATE_AND_INIT

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // allocate g_A on GPU
    CUDA_SAFE_CALL(cudaMalloc((void **)&g_A, allocSize));

    // allocate h_A_GPU and h_A_CPU on host memory
    h_A_CPU = (float *)malloc(allocSize);
    h_A_GPU = (float *)malloc(allocSize);

    printf("\nInitializing the arrays ...");

    // initialize host arrays
    initialize_Array_2D(h_A_CPU, MATRIX_SIZE, 2453);

    for (int i = 0; i < 2000; i++)
    {
        for (int j = 0; j < 2000; j++)
        {
            h_A_GPU[i * MATRIX_SIZE + j] = h_A_CPU[i * MATRIX_SIZE + j];
        }
    }

    // initialize_Array_2D(h_A_GPU, MATRIX_SIZE, 2453);
    printf("\t... done\n");

#endif

#ifdef DEBUG_PRINT
    printf("allocate all done\n");
#endif

#ifdef GPU_TIMING
    // create cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record event on default stream
    cudaEventRecord(start, 0);
#endif

#ifdef TRANSFER_TO_GPU
    CUDA_SAFE_CALL(cudaMemcpy(g_A, h_A_GPU, allocSize, cudaMemcpyHostToDevice));
#endif

#ifdef LAUNCH_KERNEL

    SOR_GPU<<<dimGrid, dimBlock>>>(g_A, OMEGA, MATRIX_SIZE, MATRIX_SIZE);

#endif

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());

#ifdef TRANSFER_RESULTS
    CUDA_SAFE_CALL(cudaMemcpy(h_A_GPU, g_A, allocSize, cudaMemcpyDeviceToHost));
#endif

#ifdef GPU_TIMING
    // record the time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // print the time
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time: %f (msec)\n", elapsed_gpu);
    // destroy
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

#ifdef DEBUG_PRINT
    printf("GPU calculation ended\n");
#endif

#ifdef CPU_TIMING
    clock_gettime(CLOCK_REALTIME, &time_start);
#endif

#ifdef COMPUTE_CPU_RESULTS
    for (i = 0; i < SOR_ITERATIONS; i++)
    {
        SOR_CPU(h_A_CPU, OMEGA, MATRIX_SIZE, MATRIX_SIZE);
    }
#endif

#ifdef CPU_TIMING
    clock_gettime(CLOCK_REALTIME, &time_stop);
    elapse_time = interval(time_start, time_stop);
    printf("\nCPU time: %f(msec)\n", (float)(elapse_time));
#endif

#ifdef DEBUG_PRINT
    printf("CPU calculation ended\n");
#endif

#ifdef COMPARE_RESULTS
    for (i = 0; i < MATRIX_SIZE; i++)
    {
        for (j = 0; j < MATRIX_SIZE; j++)
        {
            int index = i * MATRIX_SIZE + j;
            if (h_A_CPU[index] >= h_A_GPU[index] * 1.05 or h_A_CPU[index] <= h_A_GPU[index] * 0.95)
            {
                errors++;
                fprintf(f, "Mismatch at [%d,%d] GPU = %f CPU = %f\n", i, j, h_A_GPU[i * MATRIX_SIZE + j], h_A_CPU[i * MATRIX_SIZE + j]);
            }
        }
    }
#endif

#ifdef DEBUG_PRINT
    printf("results check finished\n");
#endif

#ifdef WRITE_2D_ARRAYS
    write_2d_Array_To_File(h_A_GPU, MATRIX_SIZE, MATRIX_SIZE, "GPU_output.txt");
    write_2d_Array_To_File(h_A_CPU, MATRIX_SIZE, MATRIX_SIZE, "CPU_output.txt");
#endif

    // print error numbers
    printf("Found %d errors\n", errors);

#ifdef FREE_MEMORY
    CUDA_SAFE_CALL(cudaFree(g_A));
    free(h_A_CPU);
    free(h_A_GPU);

#endif

    return 0;
}
