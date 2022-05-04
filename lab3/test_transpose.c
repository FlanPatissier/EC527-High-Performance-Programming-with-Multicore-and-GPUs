/*****************************************************************************/
// gcc -O1 -std=gnu99 -mavx test_transpose.c -lrt -o test_transpose

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define CPNS 2.4   /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */

/* We want to test a range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   40  /* coefficient of x^2 */
#define B   800  /* coefficient of x */
#define C   200 /* constant term */

#define NUM_TESTS 20

#define OPTIONS 4     // ij and ji -- need to add other methods

typedef double data_t;

/* Create abstract data type for an array */
typedef struct {
  long int rowlen;
  data_t *data;
} array_rec, *array_ptr;

/* Number of bytes in a vector */
#define VBYTES 32

/* Number of elements in a vector */
#define VSIZE (VBYTES/sizeof(data_t))



array_ptr new_array(long int rowlen);
int set_array_rowlen(array_ptr v, long int index);
long int get_array_rowlen(array_ptr v);
int init_array(array_ptr v, long int len);
void transpose(array_ptr v0, array_ptr v1);
void transpose_rev(array_ptr v0, array_ptr v1);
void transpose_sse(array_ptr v0, array_ptr v1);
void transpose4(array_ptr v0, array_ptr v1);




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


/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  data_t *data_holder;
  double wd;
  long int x, n, alloc_size;

  printf("Transpose (lab3)\n");

  wd = wakeup_delay();
  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  /* declare and initialize the arrays in memory */
  array_ptr v0 = new_array(alloc_size);  init_array(v0, alloc_size);
  array_ptr v1 = new_array(alloc_size);  init_array(v1, alloc_size);
  array_ptr v2 = new_array(alloc_size);  init_array(v2, alloc_size);
  array_ptr v3 = new_array(alloc_size);  init_array(v3, alloc_size);


  OPTION = 0;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_rowlen(v0, n);
    set_array_rowlen(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    transpose(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_rowlen(v0, n);
    set_array_rowlen(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    transpose_rev(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_rowlen(v0, n);
    set_array_rowlen(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    transpose_sse(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_rowlen(v0, n);
    set_array_rowlen(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    transpose4(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  /* output times */
  printf("size,   ij,   ji,  SSE transpose, transpose4\n");
  {
    for (int i = 0; i < x; i++) {
      printf("%ld,  ", (long)(A*i*i + B*i + C));
      for (int j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }

  printf("\n");
  printf("Wakeup delay calculated %f\n", wd);
  
} /* end main */
/*********************************/

/* Create 2D array in memory, with the specified length per dimension */
array_ptr new_array(long int rowlen)
{
  long int i;

  /* Allocate and declare header structure */
  array_ptr result = (array_ptr) malloc(sizeof(array_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->rowlen = rowlen;

  if (rowlen % VSIZE) {
    fprintf(stdout,
      "new_array: row length %ld is not a multiple of vector size (VSIZE=%d)\n"
      "to fix this, A, B and C must all be a multiple of %d\n",
      rowlen, ((int)VSIZE), ((int)VSIZE));
    exit(-1);
  }

  /* Allocate and declare array */
  if (rowlen > 0) {
    data_t *data = (data_t *) calloc(rowlen*rowlen+VSIZE, sizeof(data_t));
    if (!data) {
      /* Couldn't allocate storage */
      free((void *) result);
      fprintf(stderr, " COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                        (rowlen*rowlen+VSIZE)*sizeof(data_t));
      exit(-1);
    }
    while (((long)data) % (VSIZE * sizeof(data_t)) != 0) {
      data++;
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set length of an array. This does NOT change the amount of space allocated
   in memory, it just changes the field that controls how much data we work on
   in the transpose routines. */
int set_array_rowlen(array_ptr v, long int index)
{
  v->rowlen = index;
  return 1;
}

/* Return row length of an array */
long int get_array_rowlen(array_ptr v)
{
  return v->rowlen;
}

/* initialize 2D array with consecutive integers (converted to data_t type) */
int init_array(array_ptr v, long int len)
{
  if (len > 0) {
    v->rowlen = len;
    for (long i = 0; i < len*len; i++) {
      v->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

data_t *get_array_start(array_ptr v)
{
  return v->data;
}


/************************************/

/* transpose */
void transpose(array_ptr v0, array_ptr v1)
{
  long int get_array_rowlen(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int length = get_array_rowlen(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  for (long i = 0; i < length; i++) {
    for (long j = 0; j < length; j++) {
      data1[j*length+i] = data0[i*length+j];
    }
  }
}

/* transpose with loops interchanged */
void transpose_rev(array_ptr v0, array_ptr v1)
{
  long int get_array_rowlen(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int length = get_array_rowlen(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  for (long j = 0; j < length; j++) {
    for (long i = 0; i < length; i++) {
      data1[j*length+i] = data0[i*length+j];
    }
  }
}


void transpose_sse(array_ptr v0, array_ptr v1)
{

  long int i, j, m, n;
  __m128   m1, m2, m3, m4;
  __m128* pSrc1;
  __m128* pSrc2;

  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  pSrc1 = (__m128*)(data0);
  pSrc2 = (__m128*)(data1);

 
  long int get_array_rowlen(array_ptr v);
  long int length = get_array_rowlen(v0)/4;

      for (i = 0; i < length; ++i){
          for(j =0; j< length ; ++j){


            m1 = (*pSrc1);
            pSrc1++;
            m2 = (*pSrc1);
            pSrc1++;
            m3 = (*pSrc1);
            pSrc1++;
            m4 = (*pSrc1);
            pSrc1++;

            _MM_TRANSPOSE4_PS(m1,m2,m3,m4);

            (*pSrc2) = m1;
            pSrc2++;
            (*pSrc2) = m2;
            pSrc2++;
            (*pSrc2) = m3;
            pSrc2++;
            (*pSrc2) = m4;
            pSrc2++;
        }
      }
}


void transpose4(array_ptr v0, array_ptr v1)
{
  
  long int i, j, m, n;
  __m256d   m1, m2, m3, m4;
  __m256d  tmp3, tmp2, tmp1, tmp0;
  __m256d* pSrc1;
  __m256d* pSrc2;

  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  pSrc1 = (__m256d*)(data0);
  pSrc2 = (__m256d*)(data1);

 
  long int get_array_rowlen(array_ptr v);
  long int length = get_array_rowlen(v0)/4;

  for (i = 0; i < length; ++i){
          for(j =0; j< length ; ++j){


            m1 = (*pSrc1);
            pSrc1++;
            m2 = (*pSrc1);
            pSrc1++;
            m3 = (*pSrc1);
            pSrc1++;
            m4 = (*pSrc1);
            pSrc1++;

                                          
                                                                                 
            tmp0 = _mm256_shuffle_pd((m1),(m2), 0x0);                    
            tmp2 = _mm256_shuffle_pd((m1),(m2), 0xF);                
            tmp1 = _mm256_shuffle_pd((m3),(m4), 0x0);                    
            tmp3 = _mm256_shuffle_pd((m3),(m4), 0xF);                
                                                                                 
            m1 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);   
            m2 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);   
            m3 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);   
            m4 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);   

            (*pSrc2) = m1;
            pSrc2++;
            (*pSrc2) = m2;
            pSrc2++;
            (*pSrc2) = m3;
            pSrc2++;
            (*pSrc2) = m4;
            pSrc2++;
        }
      }

}