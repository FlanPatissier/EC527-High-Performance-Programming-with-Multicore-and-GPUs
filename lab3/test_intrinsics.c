/*
  gcc -O1 -std=gnu99 -mavx test_intrinsics.c -lm -lrt -o test_intrinsics
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define CPNS 2.4    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */

/* We want to test a range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   40  /* coefficient of x^2 */
#define B   800  /* coefficient of x */
#define C   200  /* constant term */

#define NUM_TESTS 20


#define OUTER_LOOPS 1000

#define OPTIONS 8

typedef float data_t;


void InitArray(data_t* pA, long int nSize);
void InitArray_rand(data_t* pA, long int nSize);
void ZeroArray(data_t* pA, long int nSize);
void scalar_distance(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void SSE_distance(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void AVX_distance(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void Element_Add1(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void Element_Multi1(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void Element_Add2(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void Element_Multi2(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);
void dot_product(data_t* pA1, data_t* pA2, data_t* pR, long int nSize);


/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */


/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (int i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/**************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double* var;
  int     ok;
  data_t*  pArray1;
  data_t*  pArray2;
  data_t*  pResult;
  long int nSize;
  double wd;
  long int x, n, alloc_size;
 
  printf("Hello World!  SSE Test\n");

  wd = wakeup_delay();

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  ok = posix_memalign((void**)&pArray1, 64, alloc_size*sizeof(data_t));
  ok = posix_memalign((void**)&pArray2, 64, alloc_size*sizeof(data_t));
  ok = posix_memalign((void**)&pResult, 64, alloc_size*sizeof(data_t));

  /* initialize pArray1, pArray2 */
  InitArray_rand(pArray1, alloc_size);
  InitArray_rand(pArray2, alloc_size);
  ZeroArray(pResult, alloc_size);

  OPTION = 0;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      scalar_distance(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      SSE_distance(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      AVX_distance(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      Element_Add1(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      Element_Multi1(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      Element_Add2(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      Element_Multi2(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  } 


  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot_product(pArray1, pArray2, pResult, n);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  /* output times */
  printf("size, Scalar, vecSSE, vecAVX, Element_Add, Element_Multi, dot_product\n");
  {
    for (int i = 0; i < x; i++) {
      printf("%8d, ", (A*i*i + B*i + C) * OUTER_LOOPS);
      for (int j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%8ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }

  printf("\n");
  printf("Wakeup delay calculated %f\n", wd);

} /* end main */


double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

/* initialize array to index */
void InitArray(data_t* v, long int len)
{
  for (long i = 0; i < len; i++) {
    v[i] = (data_t)(i);
  }
}

/* initialize an array with random values */
void InitArray_rand(data_t* v, long int len)
{
  double fRand(double fMin, double fMax);

  for (long i = 0; i < len; i++) {
    v[i] = (data_t)(fRand((double)(0.0),(double)(10.0)));
  }
}

/* initialize an array with 0s */
void ZeroArray(data_t* v, long int len)
{
  for (long i = 0; i < len; i++) {
    v[i] = (data_t)(0);
  }
}

/**************************************************************/

/* Simple distance calc -- non-vectorised version */
void scalar_distance(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)        // [in] size of all arrays
{
  data_t* pSource1 = pArray1;
  data_t* pSource2 = pArray2;
  data_t* pDest = pResult;
  float sqrtf(float x);

  for (long i = 0; i < nSize; i++){
    *pDest = sqrtf((*pSource1) * (*pSource1) +
                   (*pSource2) * (*pSource2)) + 0.5f;
    pSource1++;
    pSource2++;
    pDest++;
  }
}

/* Simple distance calc w/ SSE */
void SSE_distance(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{
  long nLoop = nSize/4;

  __m128   m1, m2, m3, m4;
  __m128   m0_5 = _mm_set_ps1(0.5f);

  __m128*  pSrc1 = (__m128*) pArray1;
  __m128*  pSrc2 = (__m128*) pArray2;
  __m128*  pDest = (__m128*) pResult;

  for (long i = 0; i < nLoop; i++){
    m1 = _mm_mul_ps(*pSrc1, *pSrc1);
    m2 = _mm_mul_ps(*pSrc2, *pSrc2);
    m3 = _mm_add_ps(m1,m2);
    m4 = _mm_sqrt_ps(m3);
    *pDest = _mm_add_ps(m4,m0_5);

    pSrc1++;
    pSrc2++;
    pDest++;
  }
}

/* same distance calc as in ArrayTest2, using AVX (256-bit vectors) */
void AVX_distance(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{
  long nLoop = nSize/8;

  __m256   m1, m2, m3, m4;
  __m256   m0_5 = _mm256_set1_ps(0.5f);

  __m256*  pSrc1 = (__m256*) pArray1;
  __m256*  pSrc2 = (__m256*) pArray2;
  __m256*  pDest = (__m256*) pResult;

  for (long i = 0; i < nLoop; i++){
    m1 = _mm256_mul_ps(*pSrc1, *pSrc1);
    m2 = _mm256_mul_ps(*pSrc2, *pSrc2);
    m3 = _mm256_add_ps(m1,m2);
    m4 = _mm256_sqrt_ps(m3);
    *pDest = _mm256_add_ps(m4,m0_5);

    pSrc1++;
    pSrc2++;
    pDest++;
  }
}

//Element-wise Add using vector
void Element_Add1(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{

  long nLoop = nSize/4;

  __m128   m1, m2, m3, m4;
  __m128*  pSrc1 = (__m128*) pArray1;
  __m128*  pSrc2 = (__m128*) pArray2;
  __m128*  pDest = (__m128*) pResult;

  for (long i = 0; i < nLoop; i++){

    *pDest =_mm_add_ps(*pSrc1,*pSrc2);
    pSrc1++;
    pSrc2++;
    pDest++;
  }

}

//Element-wise Multiply using vectror
void Element_Multi1(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{
  long nLoop = nSize/4;

  __m128   m1, m2, m3, m4;
  __m128*  pSrc1 = (__m128*) pArray1;
  __m128*  pSrc2 = (__m128*) pArray2;
  __m128*  pDest = (__m128*) pResult;

  for (long i = 0; i < nLoop; i++){

    *pDest =_mm_mul_ps(*pSrc1,*pSrc2);
    pSrc1++;
    pSrc2++;
    pDest++;
  }
}

void Element_Add2(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{

  long nLoop = nSize/8;

  __m256   m1, m2, m3, m4;
  __m256*  pSrc1 = (__m256*) pArray1;
  __m256*  pSrc2 = (__m256*) pArray2;
  __m256*  pDest = (__m256*) pResult;

  for (long i = 0; i < nLoop; i++){

    *pDest =_mm256_add_ps(*pSrc1,*pSrc2);
    pSrc1++;
    pSrc2++;
    pDest++;
  }

}

void Element_Multi2(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{

  long nLoop = nSize/8;

  __m256   m1, m2, m3, m4;
  __m256*  pSrc1 = (__m256*) pArray1;
  __m256*  pSrc2 = (__m256*) pArray2;
  __m256*  pDest = (__m256*) pResult;

  for (long i = 0; i < nLoop; i++){

    *pDest =_mm256_mul_ps(*pSrc1,*pSrc2);
    pSrc1++;
    pSrc2++;
    pDest++;
  }

}

//Dot Product through Intrinsics
void dot_product(data_t* pArray1,       // [in] 1st source array
                data_t* pArray2,       // [in] 2nd source array
                data_t* pResult,       // [out] result array
                long int nSize)            // [in] size of all arrays
{

  int  i, nLoop = nSize/4;

  __m128   m1, m2, m3, m4;
  const int M = 0xff;

  __m128*  pSrc1 = (__m128*) pArray1;
  __m128*  pSrc2 = (__m128*) pArray2;
  __m128*  pDest = (__m128*) pResult;

  for (i = 0; i < nLoop; i++){
    m1 = (*pSrc1);
    m2 = (*pSrc2);
    m3 = _mm_dp_ps(m1,m2, M);
    *pDest = m3;
    pSrc1++;
    pSrc2++;
    pDest++;
  }
  // printf("%8ld",pDest);

}