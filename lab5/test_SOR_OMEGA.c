/*
   gcc -O1 test_SOR_OMEGA.c -lm -o test_SOR_OMEGA

 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 240

#define MINVAL   0.0
#define MAXVAL  100.0

#define TOL 0.00001

#define START_OMEGA 0.50 /* The first OMEGA value to try */
#define OMEGA_INC 0.01   /* OMEGA increment for each O_ITERS */
#define O_ITERS 150    /* How many OMEGA values to test */

#define PER_O_TRIALS 10  /* trials per OMEGA value */

double OMEGA = START_OMEGA;

typedef double data_t;

/* Create abstract data type for a 2D array */
typedef struct {
  long int rowlen;
  data_t *data;
} arr_rec, *arr_ptr;

arr_ptr new_array(long int row_len);
int set_arr_rowlen(arr_ptr v, long int index);
long int get_arr_rowlen(arr_ptr v);
int init_array(arr_ptr v, long int row_len);
int init_array_rand(arr_ptr v, long int row_len);
int print_array(arr_ptr v);

void SOR(arr_ptr v, int *iterations);
void SOR_ji(arr_ptr v, int *iterations);
void SOR_blocked(arr_ptr v, int *iterations);

/*****************************************************************************/
int main(int argc, char *argv[])
{
  double convergence[O_ITERS][2];
  int *iterations;
  long int i, j, k;

  printf("SOR OMEGA test\n");

  /* declare and initialize the array */
  arr_ptr v0 = new_array(ARRAY_SIZE);
  iterations = (int *) malloc(sizeof(int));

  printf("Array size = %d x %d\n", ARRAY_SIZE, ARRAY_SIZE);
  OMEGA = START_OMEGA;
  for (i = 0; i < O_ITERS; i++) {
    printf("%0.2f", OMEGA);
    double acc = 0.0;
    for (j = 0; j < PER_O_TRIALS; j++) {
      init_array_rand(v0, ARRAY_SIZE);
      SOR(v0, iterations);
      acc += (double)(*iterations);
      printf(", %d", *iterations);
    }
    printf("\n");
    convergence[i][0] = OMEGA;
    convergence[i][1] = acc/(double)(PER_O_TRIALS);
    OMEGA += OMEGA_INC;
  }
  printf("\n");

  printf("OMEGA, average iterations\n");
  for (i = 0; i < O_ITERS; i++) {
    printf("%0.4f \n", convergence[i][0]);
  }

  for (i = 0; i < O_ITERS; i++) {
    printf("%0.1f \n", convergence[i][1]);
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
      printf("COULDN'T ALLOCATE %ld bytes STORAGE \n",
                                       row_len * row_len * sizeof(data_t));
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

/* initialize array with random numbers in a range */
int init_array_rand(arr_ptr v, long int row_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  /* Since we're exploring OMEGA, it is more useful to get fresh randomness
 *      every time.
 *        srandom(row_len);
 *          */
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
    for (j = 0; j < row_len; j++)
      printf("%.4f ", (data_t)(v->data[i*row_len+j]));
    printf("\n");
  }
}

data_t *get_arr_start(arr_ptr v)
{
  return v->data;
}

/************************************/

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
  long int row_len = get_arr_rowlen(v);
  data_t *data = get_arr_start(v);
  double change, mean_change = 1.0e10;   /* start with something big */
  int iters = 0;

  if (OMEGA >= 2.0) {
    printf("Skipping test because %f is too big for convergence.\n", OMEGA);
    *iterations = INT_MAX;
    return;
  } else if (OMEGA < 0.1) {
    printf("Skipping test because %f is too small for convergence.\n", OMEGA);
    *iterations = INT_MAX;
    return;
  }

  while ((mean_change/(double)(row_len*row_len)) > (double)TOL) {
    iters++;
    mean_change = 0;
    for (i = 1; i < row_len-1; i++) {
      for (j = 1; j < row_len-1; j++) {
        change = data[i * row_len + j]
          - .25 * ( data[(i-1) * row_len +  j ] +
                    data[(i+1) * row_len +  j ] +
                    data[  i   * row_len + j+1] +
                    data[  i   * row_len + j-1]);
        data[i * row_len + j] -= change * OMEGA;
        if (change < 0){
          change = -change;
        }
        mean_change += change;
      }
    }
    if (abs(data[row_len*(row_len-1)-2]) > 10.0*(MAXVAL - MINVAL)) {
      printf("PROBABLY DIVERGENCE iter = %ld\n", iters);
      break;
    }
  }
  *iterations = iters;
}
