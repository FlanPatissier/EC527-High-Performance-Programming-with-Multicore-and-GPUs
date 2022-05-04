/***************************************************************************

  gcc -O1 test_branch.c -lrt -o test_branch

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define CPNS 2.4    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */

/* We want to test a range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   5  /* coefficient of x^2 */
#define B   100  /* coefficient of x */
#define C   50  /* constant term */

#define NUM_TESTS 20   /* Number of different sizes to test */

#define OUTER_LOOPS 2000

#define OPTIONS 4
#define IDENT 1.0
#define OP *

typedef float data_t;

/* Create abstract data type for an array in memory */
typedef struct {
  long int len;
  data_t *data;
} array_rec, *array_ptr;

array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
int set_array_length(array_ptr v, long int index);
int init_array_unpred(array_ptr v, long int len);
double fRand(double fMin, double fMax);
void branch1(array_ptr v0, array_ptr v1, array_ptr v2, long int outer_limit);
void branch2(array_ptr v0, array_ptr v1, array_ptr v2, long int outer_limit);


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


/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wd;
  long int x, n, alloc_size;

  printf("Branch optimisation examples\n");

  wd = wakeup_delay();
  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  printf("Testing 4 different ways, on arrays of %d sizes from %d to %ld\n",
    NUM_TESTS, C, alloc_size);

  /* create array data structures */
  array_ptr v0 = new_array(alloc_size);
  array_ptr v1 = new_array(alloc_size);
  array_ptr v2 = new_array(alloc_size);

  /* execute and time all options  */

  OPTION = 0;
  init_array_pred(v0, alloc_size);
  init_array_pred(v1, alloc_size);
  printf("testing option %d: branch1() on predictable data\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    set_array_length(v2, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    branch1(v0, v1, v2, OUTER_LOOPS);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  init_array_unpred(v0, alloc_size);
  init_array_unpred(v1, alloc_size);
  printf("testing option %d: branch1() on unpredictable data\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    set_array_length(v2, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    branch1(v0, v1, v2, OUTER_LOOPS);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  init_array_pred(v0, alloc_size);
  init_array_pred(v1, alloc_size);
  printf("testing option %d: branch2() on predictable data\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    set_array_length(v2, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    branch2(v0, v1, v2, OUTER_LOOPS);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  init_array_unpred(v0, alloc_size);
  init_array_unpred(v1, alloc_size);
  printf("testing option %d: branch2() on unpredictable data\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    set_array_length(v2, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    branch2(v0, v1, v2, OUTER_LOOPS);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  /* output times */
  printf("\n");
  printf("size, pred.branch1, unpred.branch1, pred.branch2, unpred.branch2\n");
  {
    int i, j;
    for (i = 0; i < x; i++) {
      printf("%ld,  ", (long)((A*i*i + B*i + C) * OUTER_LOOPS));
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int) ((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }

  printf("\n");
  printf("Wakeup delay calculated %f\n", wd);

  return 0;
} /* end main */

/**********************************************/
/* Create a memory array of a specified length */
array_ptr new_array(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  array_ptr result = (array_ptr) malloc(sizeof(array_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  } else {
    result->data = NULL;
  }

  return result;
}

/* Retrieve an array element and store at dest.
   Return 0 (out of bounds) or 1 (successful)
*/
int get_array_element(array_ptr v, long int index, data_t *dest)
{
  if (index < 0 || index >= v->len) return 0;
  *dest = v->data[index];
  return 1;
}

/* Return length of an array */
long int get_array_length(array_ptr v)
{
  return v->len;
}

/* Set length of an array. This does NOT change the amount of memory
   allocated, it just changes the field that controls how much data is
   accessed when doing one of the tests. */
int set_array_length(array_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize an array with a "predictable" pattern */
int init_array_pred(array_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) {
      v->data[i] =50  ; /* Modify this line!! */
    }
    return 1;
  } else {
    return 0;
  }
}



//  double quasi_random = 0; /* global */

/* initialize an array with an "unpredictable" pattern */
int init_array_unpred(array_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) {
      v->data[i] = fRand(100, 5000) ; /* Modify this line!! */
    }
    return 1;
  } else {
    return 0;
  }
}

data_t *get_array_start(array_ptr v)
{
  return v->data;
}

/*************************************************/

/*  HERE IS A RANDOM NUMBER FUNCTION YOU MAY FIND USEFUL  */

/* Returns a random number in the range [fMin, fMax) */
double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

/*************************************************/
/* branch1:  test branch, based on example in B&O 5.11
 * For each element i in arrays v0 and v1, write the
 * larger into element i of v2. */
void branch1(array_ptr v0, array_ptr v1, array_ptr v2, long int outer_limit)
{
  long int i, j;
  long int length = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  data_t *data2 = get_array_start(v2);
  data_t acc = (data_t)(0);

  for(j=0; j<outer_limit; j++) {
    for (i = 0; i < length; i++) {
      if (data0[i] > data1[i]) data2[i] = data0[i];
      else data2[i] = data1[1];
    }
  }
}

/*************************************************/
/* branch2:  test branch, based on example in B&O 5.11
 * For each element i in arrays v0 and v1, write the
 * larger into element i of v2. */
void branch2(array_ptr v0, array_ptr v1, array_ptr v2, long int outer_limit)
{
  long int i, j;
  long int length = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  data_t *data2 = get_array_start(v2);
  data_t acc = (data_t)(0);

  for(j=0; j<outer_limit; j++) {
    for (i = 0; i < length; i++) {
      data2[i] = (data0[i] > data1[i]) ? data0[i] : data1[i];
    }
  }
}