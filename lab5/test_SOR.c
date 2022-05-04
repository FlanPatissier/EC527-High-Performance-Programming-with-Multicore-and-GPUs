/*
   gcc -O1 -std=gnu11 test_SOR.c -lpthread -lrt -lm -o test_SOR

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
# include "apple_pthread_barrier.h"
#endif /* __APPLE__ */

#define CPNS 2.4    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define GHOST 2   /* 2 extra rows/columns for "ghost zone". */

#define A   16   /* coefficient of x^2 */
#define B   16  /* coefficient of x */
#define C   32  /* constant term */

#define NUM_TESTS 10

/* A, B, and C needs to be a multiple of your BLOCK_SIZE,
 *    total array size will be (GHOST + Ax^2 + Bx + C) */

#define BLOCK_SIZE 4     // TO BE DETERMINED

#define OPTIONS 6 

#define MINVAL   0.0
#define MAXVAL  10.0

#define TOL 0.00001
#define OMEGA 1.90       // TO BE DETERMINED

typedef double data_t;

typedef struct {
  long int rowlen;
  data_t *data;
} arr_rec, *arr_ptr;

/* Prototypes */
arr_ptr new_array(long int row_len);
int set_arr_rowlen(arr_ptr v, long int index);
long int get_arr_rowlen(arr_ptr v);
int init_array(arr_ptr v, long int row_len);
int init_array_rand(arr_ptr v, long int row_len);
int print_array(arr_ptr v);

void SOR(arr_ptr v, int *iterations);
void SOR_redblack(arr_ptr v, int *iterations);
void SOR_ji(arr_ptr v, int *iterations);
void SOR_blocked_4(arr_ptr v, int *iterations);
void SOR_blocked_8(arr_ptr v, int *iterations);
void SOR_blocked_16(arr_ptr v, int *iterations);


/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
 *   As described in the clock_gettime manpage (type "man clock_gettime" at the
 *     shell prompt), a "timespec" is a structure that looks like this:
 *      
 *              struct timespec {
 *                        time_t   tv_sec;   // seconds
 *                                  long     tv_nsec;  // and nanoseconds
 *                                          };
 *                                           */

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
 *      This method does not require adjusting a #define constant
 *
 *        How to use this method:
 *
 *              struct timespec time_start, time_stop;
 *                    clock_gettime(CLOCK_REALTIME, &time_start);
 *                          // DO SOMETHING THAT TAKES TIME
 *                                clock_gettime(CLOCK_REALTIME, &time_stop);
 *                                      measurement = interval(time_start, time_stop);
 *
 *                                       */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  int convergence[OPTIONS][NUM_TESTS];
  int *iterations;

  long int x, n;
  long int alloc_size;

  x = NUM_TESTS-1;
  alloc_size = GHOST + A*x*x + B*x + C;

  printf("SOR serial variations \n");

  printf("OMEGA = %0.2f\n", OMEGA);

  /* declare and initialize the array */
  arr_ptr v0 = new_array(alloc_size);

  /* Allocate space for return value */
  iterations = (int *) malloc(sizeof(int));

  OPTION = 0;
  printf("OPTION=%d (normal serial SOR)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (serial SOR_redblack)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_redblack(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (serial SOR_ji)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_ji(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (serial SOR_blocked)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_blocked_4(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (serial SOR_blocked)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_blocked_8(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }

  OPTION++;
  printf("OPTION=%d (serial SOR_blocked)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR_blocked_16(v0, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = *iterations;
  }


  printf("All times are in cycles (if CPNS is set correctly in code)\n");
  printf("\n");
  printf("size, SOR time, SOR iters, red/black time, red/black iters, SOR_ji time, SOR_ji iters, SOR_blocked time, SOR_blocked iters\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%4d", A*i*i + B*i + C);
      for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
        printf(", %10.4g", (double)CPNS * 1.0e9 * time_stamp[OPTION][i]);
        printf(", %4d", convergence[OPTION][i]);
      }
      printf("\n");
    }
  }

} /* end main */

/*********************************/

/* Create 2D array of specified length per dimension */
arr_ptr new_array(long int row_len)
{
  long int i;

  /* Allocate and declare header structure */
  arr_ptr result = (arr_ptr) malloc(sizeof(arr_rec));
  if (!result) {
    return NULL;  /* Couldn't allocate storage */
  }
  result->rowlen = row_len;

  /* Allocate and declare array */
  if (row_len > 0) {
    data_t *data = (data_t *) calloc(row_len*row_len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE STORAGE \n", result->rowlen);
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set row length of array */
int set_arr_rowlen(arr_ptr v, long int row_len)
{
  v->rowlen = row_len;
  return 1;
}

/* Return row length of array */
long int get_arr_rowlen(arr_ptr v)
{
  return v->rowlen;
}

/* initialize 2D array with incrementing values (0.0, 1.0, 2.0, 3.0, ...) */
int init_array(arr_ptr v, long int row_len)
{
  long int i;

  if (row_len > 0) {
    v->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      v->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

/* initialize array with random data */
int init_array_rand(arr_ptr v, long int row_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  /* Since we're comparing different algorithms (e.g. blocked, threaded
 *      with stripes, red/black, ...), it is more useful to have the same
 *           randomness for any given array size */
  srandom(row_len);
  if (row_len > 0) {
    v->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++) {
      v->data[i] = (data_t)(fRand((double)(MINVAL),(double)(MAXVAL)));
    }
    return 1;
  }
  else return 0;
}

/* print all elements of an array */
int print_array(arr_ptr v)
{
  long int i, j, row_len;

  row_len = v->rowlen;
  printf("row length = %ld\n", row_len);
  for (i = 0; i < row_len; i++) {
    for (j = 0; j < row_len; j++) {
      printf("%.4f ", (data_t)(v->data[i*row_len+j]));
    }
    printf("\n");
  }
}

data_t *get_array_start(arr_ptr v)
{
  return v->data;
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

/************************************/

/* SOR */
void SOR(arr_ptr v, int *iterations)
{
  long int i, j;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (i = 1; i < rowlen-1; i++) {
      for (j = 1; j < rowlen-1; j++) {
        change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                          data[(i+1)*rowlen+j] +
                                          data[i*rowlen+j+1] +
                                          data[i*rowlen+j-1]);
        data[i*rowlen+j] -= change * OMEGA;
        if (change < 0){
          change = -change;
        }
        total_change += change;
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR() done after %d iters\n", iters);
}

/* SOR red/black */
void SOR_redblack(arr_ptr v, int *iterations)
{
  int i, j, redblack;
  long int ti;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;

  ti = 0;
  redblack = 0;
  /* The while condition here tests the tolerance limit *only* when
 *      redblack is 0, which ensures we exit only after having done a
 *           full update (red + black) */
  while ((redblack == 1)
        || ((total_change/(double)(rowlen*rowlen)) > (double)TOL) )
  {
    /* Reset sum of total change only when starting a black scan. */
    if (redblack == 0) {
      total_change = 0;
    }
    for (i = 1; i < rowlen-1; i++) {
      /* The j loop needs to start at j=1 on row 0 and all even rows,
 *          and start at j=2 on odd rows; but when redblack is true it does
 *                   just the opposite; and it always increments by 2. */
      for (j = 1 + ((i^redblack)&1); j < rowlen-1; j+=2) {
        change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                          data[(i+1)*rowlen+j] +
                                          data[i*rowlen+j+1] +
                                          data[i*rowlen+j-1]);
        data[i*rowlen+j] -= change * OMEGA;
        if (change < 0) {
          change = -change;
        }
        total_change += change;
        ti++;
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
      break;
    }
    redblack ^= 1;
    iters++;
  }
  /* A "red scan" only updates half of the array, and likewise for a
 *      "black scan"; so we need to divide iters by 2 to convert our count of
 *           "reds+blacks" to a count of "full scans" */
  iters /= 2;
  *iterations = iters;
  printf("    SOR_redblack() done after %d iters\n", iters);
  /* printf("ti == %ld, per iter %ld\n", ti, ti/iters); */
} /* End of SOR_redblack */

/* SOR with reversed indices */
void SOR_ji(arr_ptr v, int *iterations)
{
  long int i, j;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (j = 1; j < rowlen-1; j++) {
      for (i = 1; i < rowlen-1; i++) {
        change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                          data[(i+1)*rowlen+j] +
                                          data[i*rowlen+j+1] +
                                          data[i*rowlen+j-1]);
        data[i*rowlen+j] -= change * OMEGA;
        if (change < 0){
          change = -change;
        }
        total_change += change;
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR_ji: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR_ji() done after %d iters\n", iters);
}

/* SOR w/ blocking */
void SOR_blocked_4(arr_ptr v, int *iterations)
{
  long int i, j, ii, jj;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;
  int iters = 0;
  int k;

  if ((rowlen-2) % (BLOCK_SIZE)) {
    fprintf(stderr,
"SOR_blocked: Total array size must be 2 more than a multiple of BLOCK_SIZE\n"
"(because the top/right/left/bottom rows are not scanned)\n"
"Make sure all coefficients A, B, and C are multiples of %d\n", BLOCK_SIZE);
    exit(-1);
  }

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (ii = 1; ii < rowlen-1; ii+=BLOCK_SIZE) {
      for (jj = 1; jj < rowlen-1; jj+=BLOCK_SIZE) {
        for (i = ii; i < ii+BLOCK_SIZE; i++) {
          for (j = jj; j < jj+BLOCK_SIZE; j++) {
            change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                              data[(i+1)*rowlen+j] +
                                              data[i*rowlen+j+1] +
                                              data[i*rowlen+j-1]);
            data[i*rowlen+j] -= change * OMEGA;
            if (change < 0){
              change = -change;
            }
            total_change += change;
          }
        }
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR_blocked: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR_blocked() done after %d iters\n", iters);
} /* End of SOR_blocked */

/* SOR w/ blocking */
void SOR_blocked_8(arr_ptr v, int *iterations)
{
  long int i, j, ii, jj;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;
  int iters = 0;
  int k;

  if ((rowlen-2) % (8)) {
    fprintf(stderr,
"SOR_blocked: Total array size must be 2 more than a multiple of BLOCK_SIZE\n"
"(because the top/right/left/bottom rows are not scanned)\n"
"Make sure all coefficients A, B, and C are multiples of %d\n", 8);
    exit(-1);
  }

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (ii = 1; ii < rowlen-1; ii+=8) {
      for (jj = 1; jj < rowlen-1; jj+=8) {
        for (i = ii; i < ii+8; i++) {
          for (j = jj; j < jj+8; j++) {
            change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                              data[(i+1)*rowlen+j] +
                                              data[i*rowlen+j+1] +
                                              data[i*rowlen+j-1]);
            data[i*rowlen+j] -= change * OMEGA;
            if (change < 0){
              change = -change;
            }
            total_change += change;
          }
        }
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR_blocked: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR_blocked() done after %d iters\n", iters);
} /* End of SOR_blocked */

/* SOR w/ blocking */
void SOR_blocked_16(arr_ptr v, int *iterations)
{
  long int i, j, ii, jj;
  long int rowlen = get_arr_rowlen(v);
  data_t *data = get_array_start(v);
  double change, total_change = 1.0e10;
  int iters = 0;
  int k;

  if ((rowlen-2) % (16)) {
    fprintf(stderr,
"SOR_blocked: Total array size must be 2 more than a multiple of BLOCK_SIZE\n"
"(because the top/right/left/bottom rows are not scanned)\n"
"Make sure all coefficients A, B, and C are multiples of %d\n", 16);
    exit(-1);
  }

  while ((total_change/(double)(rowlen*rowlen)) > (double)TOL) {
    iters++;
    total_change = 0;
    for (ii = 1; ii < rowlen-1; ii+=16) {
      for (jj = 1; jj < rowlen-1; jj+=16) {
        for (i = ii; i < ii+16; i++) {
          for (j = jj; j < jj+16; j++) {
            change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                              data[(i+1)*rowlen+j] +
                                              data[i*rowlen+j+1] +
                                              data[i*rowlen+j-1]);
            data[i*rowlen+j] -= change * OMEGA;
            if (change < 0){
              change = -change;
            }
            total_change += change;
          }
        }
      }
    }
    if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("SOR_blocked: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  *iterations = iters;
  printf("    SOR_blocked() done after %d iters\n", iters);
} /* End of SOR_blocked */

