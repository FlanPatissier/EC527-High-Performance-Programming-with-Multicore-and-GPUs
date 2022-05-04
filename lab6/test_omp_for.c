/********************************************************************

 gcc -O1 -fopenmp test_omp_for.c -lrt -lm -o test_omp_for
 OMP_NUM_THREADS=1 ./test_omp_for

This program has three functions, all variations of simple FOR loop
program with independent iterations.

 1. Compute bound -- lots of comptutation on each array element

 2. Memory bound -- multiple memory references, including pointer
                    following, for each array element

 3. Overhead bound -- not much work per array element

Each function also has a serial baseline.

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* We do *not* use CPNS because when multiple cores are each executing with
   their own clocks, sometimes overlapping in time, measuring "how many
   cycles" a program takes is less meaningful. */

#define A   1  /* coefficient of x^2 */
#define B   1  /* coefficient of x */
#define C  4  /* constant term */

#define NUM_TESTS 20

#define OPTIONS 6

#define INIT_LOW -10.0
#define INIT_HIGH 10.0

typedef long int data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int rowlen;
  data_t *data;
} matrix_rec, *matrix_ptr;

/* Prototypes */
double fRand(double fMin, double fMax);
matrix_ptr new_matrix(long int len);
int set_matrix_rowlen(matrix_ptr m, long int index);
long int get_matrix_rowlen(matrix_ptr m);
int init_matrix(matrix_ptr m, long int len);
int zero_matrix(matrix_ptr m, long int len);
int init_matrix_rand(matrix_ptr m, long int rowlen);
int init_matrix_rand_ptr(matrix_ptr m, long int rowlen);
void cpu_bound_serial(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void omp_cpu_bound(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mem_bound_serial(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d);
void omp_mem_bound(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d);
void overhead_bound_serial(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void omp_overhead_bound(matrix_ptr a, matrix_ptr b, matrix_ptr c);


/* This define is only used if you do not set the environment variable
   OMP_NUM_THREADS as instructed above, and if OpenMP also does not
   automatically detect the hardware capabilities.

   If you have a machine with lots of cores, you may wish to test with
   more threads, but make sure you also include results for THREADS=4
   in your report. */
#define THREADS 4

void detect_threads_setting()
{
  long int i, ognt;
  char * env_ONT;

  /* Find out how many threads OpenMP thinks it is wants to use */
#pragma omp parallel for
  for(i=0; i<1; i++) {
    ognt = omp_get_num_threads();
  }

  printf("omp's default number of threads is %d\n", ognt);

  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0) {
    if (THREADS != ognt) {
      printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }

  omp_set_num_threads(ognt);

  /* Once again ask OpenMP how many threads it is going to use */
#pragma omp parallel for
  for(i=0; i<1; i++) {
    ognt = omp_get_num_threads();
  }
  printf("Using %d threads for OpenMP\n", ognt);
}



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
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
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


/*********************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double final_answer;
  long int x, n, i, j, alloc_size;

  printf("OpenMP parallel loop tests\n");

  final_answer = wakeup_delay();
  detect_threads_setting();

  /* declare and initialize the matrices */
  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;
  matrix_ptr a0 = new_matrix(alloc_size);
  init_matrix_rand(a0, alloc_size);
  matrix_ptr b0 = new_matrix(alloc_size);
  init_matrix_rand(b0, alloc_size);
  matrix_ptr c0 = new_matrix(alloc_size);
  zero_matrix(c0, alloc_size);
  matrix_ptr d0 = new_matrix(alloc_size);
  init_matrix_rand_ptr(d0, alloc_size);

  for (OPTION = 0; OPTION<OPTIONS; OPTION++) {
    printf("Doing OPTION=%d...\n", OPTION);
    for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
      set_matrix_rowlen(a0, n);
      set_matrix_rowlen(b0, n);
      set_matrix_rowlen(c0, n);
      set_matrix_rowlen(d0, n);
      clock_gettime(CLOCK_REALTIME, &time_start);
      switch(OPTION) {
        case 0:      cpu_bound_serial(a0, b0, c0);     break;
        case 1:         omp_cpu_bound(a0, b0, c0);     break;
        case 2:      mem_bound_serial(a0, b0, c0, d0); break;
        case 3:         omp_mem_bound(a0, b0, c0, d0); break;
        case 4: overhead_bound_serial(a0, b0, c0);     break;
        case 5:    omp_overhead_bound(a0, b0, c0);     break;
        default: break;
      }
      clock_gettime(CLOCK_REALTIME, &time_stop);
      time_stamp[OPTION][x] = interval(time_start, time_stop);
      printf("  iter %d done\r", x); fflush(stdout);
    }
    printf("\n");
  }


  printf("\nAll times are in seconds\n");
  printf("rowlen, cpu_bound_serial, omp_cpu_bound, mem_bound_serial, omp_mem_bound, ooverhead_bound_serial, omp_overhead_bound\n");
  {
    int i, j;
    for (i = 0; i < x; i++) {
      printf("%4ld",  A*i*i + B*i + C);
      for (j = 0; j < OPTIONS; j++) {
        printf(",%10.4g", time_stamp[j][i]);
      }
      printf("\n");
    }
  }
  printf("\n");
  printf("Initial delay was calculating: %g \n", final_answer);

  return 0;
} /* end main */

/**********************************************/

/* Create matrix of specified row length */
matrix_ptr new_matrix(long int rowlen)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->rowlen = rowlen;

  /* Allocate and declare array */
  if (rowlen > 0) {
    data_t *data = (data_t *) calloc(rowlen*rowlen, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                  rowlen * rowlen * sizeof(data_t) );
      exit(-1);
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, long int index)
{
  m->rowlen = index;
  return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m)
{
  return m->rowlen;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int rowlen)
{
  long int i;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen*rowlen; i++) {
      m->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / (double)(RAND_MAX);
  return fMin + f * (fMax - fMin);
}

/* initialize matrix to rand */
int init_matrix_rand(matrix_ptr m, long int rowlen)
{
  long int i;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen*rowlen; i++) {
      m->data[i] = (data_t)(fRand(INIT_LOW, INIT_HIGH));
    }
    return 1;
  }
  else return 0;
}

int init_matrix_rand_ptr(matrix_ptr m, long int rowlen)
{
  long int i;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen*rowlen; i++) {
      m->data[i] = (data_t)(fRand(0, (double)(i)));
    }
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int rowlen)
{
  long int i,j;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen*rowlen; i++)
      m->data[i] = 0;
    return 1;
  }
  else return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}


/*************************************************/

/* CPU bound baseline */
void cpu_bound_serial(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < rowlen*rowlen; i++) {
    c0[i] = (data_t)(sqrt(cos(exp((double)(a0[i])))));
  }
}

/* CPU bound openmp */
void omp_cpu_bound(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

#pragma omp parallel for
  for (i = 0; i < rowlen * rowlen; i++) {
    c0[i] = (data_t)(sqrt(cos(exp((double)(a0[i])))));
  }
}

/* memory bound baseline */
void mem_bound_serial(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d)
{
  long int i, j, k;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t *d0 = get_matrix_start(d);

  for (i = 1; i < (rowlen * rowlen) - 1; i++) {
    c0[i] = a0[d0[i]]+b0[d0[i]]+c0[d0[i]]+d0[d0[i]];
  }
}

/* memory bound openmp */
void omp_mem_bound(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d)
{
  long int i, j, k;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t *d0 = get_matrix_start(d);

#pragma omp parallel for
  for (i = 1; i < (rowlen * rowlen) - 1; i++) {
    c0[i] = a0[d0[i]]+b0[d0[i]]+c0[d0[i]]+d0[d0[i]];
  }
}

/* overhead bound baseline */
void overhead_bound_serial(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 1; i < (rowlen * rowlen) - 1; i++) {
    c0[i] = a0[i];
  }
}

/* overhead bound openmp */
void omp_overhead_bound(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

#pragma omp parallel for
  for (i = 1; i < (rowlen * rowlen) - 1; i++) {
    c0[i] = a0[i];
  }
}
