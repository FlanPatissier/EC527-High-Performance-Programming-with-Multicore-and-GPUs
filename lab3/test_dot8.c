/*****************************************************************************

   gcc -O1 -std=gnu99 -mavx test_dot8.c -lrt -lm -o test_dot

 dot4    -- baseline scalar
 dot5    -- scalar unrolled by 2
 dot6_2  -- scalar unrolled by 2 w/ 2 accumulators
 dot6_5  -- scalar unrolled by 5 w/ 5 accumulators
 dot8    -- vector
 dot8_2  -- vector w/ 2 accumulators
 dot8_4  -- vector w/ 4 accumulators          TO BE WRITTEN
 dot8_8  -- vector w/ 4 accumulators          TO BE WRITTEN

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

/* >>>>> PLEASE USE LAB MACHINES FOR THIS ASSIGNMENT <<<<< */
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define CPNS 3    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */


/* We want to test a range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   40  /* coefficient of x^2 */
#define B   800  /* coefficient of x */
#define C   200  /* constant term */

#define NUM_TESTS 20


#define OUTER_LOOPS 1000

#define OPTIONS 8       // MAKE 8 AFTER ADDING dot8_4, dot8_8
#define IDENT 1.0

typedef float data_t;

/* Create abstract data type for vector */
typedef struct {
  long int len;
  data_t *data;
} array_rec, *array_ptr;

/* Number of bytes in a vector */
#define VBYTES 32

/* Number of elements in a vector */
#define VSIZE (VBYTES/sizeof(data_t))

typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));
typedef union {
  vec_t v;
  data_t d[VSIZE];
} pack_t;



int clock_gettime(clockid_t clk_id, struct timespec *tp);
array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
int set_array_length(array_ptr v, long int index);
int init_array(array_ptr v, long int len);
int init_array_rand(array_ptr v, long int len);
double fRand(long range);

void dot4(array_ptr v0, array_ptr v1, data_t *dest);
void dot5(array_ptr v0, array_ptr v1, data_t *dest);
void dot6_2(array_ptr v0, array_ptr v1, data_t *dest);
void dot6_5(array_ptr v0, array_ptr v1, data_t *dest);
void dot8(array_ptr v0, array_ptr v1, data_t *dest);
void dot8_2(array_ptr v0, array_ptr v1, data_t *dest);
 void dot8_4(array_ptr v0, array_ptr v1, data_t *dest);
 void dot8_8(array_ptr v0, array_ptr v1, data_t *dest);


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
  data_t result_values[OPTIONS][NUM_TESTS];

  double wd;
  long int x, n, alloc_size;

  __m256 PLEASE_USE_LAB_MACHINES_FOR_THIS_ASSIGNMENT;

  printf("dot product examples\n");

  wd = wakeup_delay();
  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  /* declare and initialize the array structures */
  array_ptr v0 = new_array(alloc_size);
  init_array(v0, alloc_size);
  array_ptr v1 = new_array(alloc_size);
  init_array(v1, alloc_size);
  data_holder = (data_t *) malloc(sizeof(data_t));

  /* For the question that asks you to debug dot2_8(), change this
     to "if (1)" to enable this test */
  if (0) {
    n = 10;
    set_array_length(v0, n);
    set_array_length(v1, n);
    dot4(v0, v1, data_holder);
    printf("dot4(v0, v1, %ld) == %g\n", n, *data_holder);
    dot8(v0, v1, data_holder);
    printf("dot8(v0, v1, %ld) == %g\n", n, *data_holder);
    dot8_2(v0, v1, data_holder);
    printf("dot8_2(v0, v1, %ld) == %g\n", n, *data_holder);
    exit(0);
  }

  /* Now that we have done the easy test, reinitialise the arrays with
     sort of random contents, harder to debug but more realistic */
  init_array_rand(v0, alloc_size);
  init_array_rand(v0, alloc_size);

  /* execute and time all options */
  OPTION = 0;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot4(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot5(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot6_2(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot6_5(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot8(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot8_2(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
  
  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot8_4(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }
  
  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    set_array_length(v1, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      dot8_8(v0, v1, data_holder);
    }
    result_values[OPTION][x] = *data_holder;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  printf("\n");

  /* output times */
  printf("Computed dot products:\n");
  printf("size, dot4, dot5, dot6_2, dot6_5, dot8, dot8_2\n");
  {
    for (int i = 0; i < x; i++) {
      printf("%5ld,  ", (long)(A*i*i + B*i + C));
      for (int j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%10.5g", result_values[j][i]);
      }
      printf("\n");
    }
  }
  printf("\n");

  printf("Execution times (in estimated CPU cycles):\n");
  printf("size*OUTER_LOOPS, dot4, dot5, dot6_2, dot6_5, dot8, dot8_2, dot8_4, dot8_8\n");
  {
    for (int i = 0; i < x; i++) {
      printf("%8ld,  ", (long)((A*i*i + B*i + C) * OUTER_LOOPS));
      for (int j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%8ld", (long int) ((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }
  printf("\n");

  printf("\n");
  printf("Wakeup delay calculated %f\n", wd);

} /* end main */

/**********************************************/
/* Create a 1-D array of the specified length */
array_ptr new_array(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  array_ptr result = (array_ptr) malloc(sizeof(array_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc((len + VSIZE), sizeof(data_t));
    if (!data) {
      /* Couldn't allocate storage */
      free((void *) result);
      fprintf(stderr, " COULDN'T ALLOCATE %ld BYTES STORAGE \n",
        (len+VSIZE)*sizeof(data_t));
      exit(-1);
    }
    /* Force proper alignment */
    while (((long)data) % (VSIZE * sizeof(data_t)) != 0) data++;
    result->data = data;
  }
  else result->data = NULL;

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

/* Set the length field of an array; this does NOT change the
   amount of memory allocated, just the "length" field which
   is used to control hw much of the array is used for
   the various computations. */
int set_array_length(array_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize an array of data_t */
int init_array(array_ptr v, long int len)
{
  if (len > 0) {
    v->len = len;
    for (long i = 0; i < len; i++) {
      v->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

/* initialize an array with random values (converted to type data_t) */
int init_array_rand(array_ptr v, long int len)
{
  if (len > 0) {
    v->len = len;
    for (long i = 0; i < len; i++) {
      v->data[i] = (data_t)(fRand(i+3));
    }
    return 1;
  }
  else return 0;
}

data_t *get_array_start(array_ptr v)
{
  return v->data;
}

/* Generate quasi-random numbers from 0 to "range" */
double fRand(long range)
{
  return (double)random() * range / RAND_MAX;
}

/*************************************************/
/* dot4:  Baseline scalar dot product*/
void dot4(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int length = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  data_t acc = (data_t)(0);

  for (long i = 0; i < length; i++) {
    acc += data0[i] * data1[i];
  }
  // printf("%10.5g",acc);

  *dest = acc;
} /* End of dot4 */

/* dot5:  Unroll loop by 2 */
void dot5(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int length = get_array_length(v0);
  long int limit = length - 1;
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  data_t acc = (data_t)(0);

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=2) {
    acc += data0[i] * data1[i] + data0[i+1] * data1[i+1];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc += data0[i] * data1[i];
  }
  *dest = acc;
} /* End of dot5 */

/* dot6_2:  Unroll loop by 2, 2 accumulators
 * Example of --> parallelism */
void dot6_2(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int length = get_array_length(v0);
  long int limit = length - 1;
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  data_t acc0 = (data_t)(0);
  data_t acc1 = (data_t)(0);

  /* Combine two elements at a time w/ 2 acculators */
  for (i = 0; i < limit; i+=2) {
    acc0 += data0[i] * data1[i];
    acc1 += data0[i+1] * data1[i+1];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 += data0[i] * data1[i];
  }
  *dest = acc0 + acc1;
} /* End of dot6_2 */

/* dot6_5:  Unroll loop by 5, 5 accumulators */
void dot6_5(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int length = get_array_length(v0);
  long int limit = length - 4;
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  data_t acc0 = (data_t)(0);
  data_t acc1 = (data_t)(0);
  data_t acc2 = (data_t)(0);
  data_t acc3 = (data_t)(0);
  data_t acc4 = (data_t)(0);

  /* Combine 5 elements at a time w/ 5 acculators */
  for (i = 0; i < limit; i+=5) {
    acc0 += data0[i] * data1[i];
    acc1 += data0[i+1] * data1[i+1];
    acc2 += data0[i+2] * data1[i+2];
    acc3 += data0[i+3] * data1[i+3];
    acc4 += data0[i+4] * data1[i+4];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 += data0[i] * data1[i];
  }
  *dest = acc0 + acc1 + acc2 + acc3 + acc4;
} /* End of dot6_5 */

/* dot8:  Vector */
void dot8(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int len = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  vec_t accum;
  data_t result = 0;

  // data_t result = (data_t)(0);
  pack_t xfer;

  /* initialize accum entries to 0 */
  for (i = 0; i < VSIZE; i++) {
    xfer.d[i] = (data_t)(0);
  }
  accum = xfer.v;

  /* i counts the number of elements done so far */
  i = 0;

  /* Single step until we have memory alignment */
  while ((((long) data0) % VBYTES) && (i < len)) {
    result += *data0++ * *data1++;
    i++;
  }

  /* Step through data with VSIZE-way parallelism */
  while (i <= (len - VSIZE)) {
    vec_t v0chunk = *((vec_t *) data0);
    vec_t v1chunk = *((vec_t *) data1);
    accum = accum + (v0chunk * v1chunk);
    data0 += VSIZE;
    data1 += VSIZE;
    i += VSIZE;
  }

  /* Single step through remaining elements */
  while (i < len) {
    result += *data0++ * *data1++;
    i++;
  }

  /* Combine elements of accumulator vector */
  xfer.v = accum;
  for (i = 0; i < VSIZE; i++) {
    result += xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of dot8 */

/* dot8_2:  Vector, 2 accumulators */
void dot8_2(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int cnt = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  vec_t accum0;
  vec_t accum1;
  data_t result = (data_t)(0);
  pack_t xfer;

  /* initialize accum entries to 0 */
  for ( i = 0; i < VSIZE; i++) {
    xfer.d[i] = (data_t)(0);
  }
  accum0 = xfer.v;
  accum1 = xfer.v;

  /* Single step until we have memory alignment */
  while ((((long) data0) % VBYTES) &&  cnt) {
    result += *data0++ * *data1++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism and dual accumulators */
  while (cnt >= 2*VSIZE) {
    vec_t v0chunk0 = *((vec_t *) data0);
    vec_t v0chunk1 = *((vec_t *) (data0+VSIZE));
    vec_t v1chunk0 = *((vec_t *) data1);
    vec_t v1chunk1 = *((vec_t *) (data1+VSIZE));
    accum0 = accum0 + (v0chunk0 * v1chunk0);
    accum1 = accum1 + (v0chunk1 * v1chunk1);
    data0 += 2*VSIZE;
    data1 += 2*VSIZE;
    cnt -= 2*VSIZE;
  }

  /* Single step through remaining elements */
  while (cnt>0) {
    result += *data0++ * *data1++;
    cnt--;
  }

  /* Combine elements of accumulator vector */
  xfer.v = accum0 + accum1;
  for ( i = 0; i < VSIZE; i++) {
    result += xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of dot8_2 */

/* dot8_4:  Vector, 4 accumulators */
void dot8_4(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int cnt = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  vec_t accum0;
  vec_t accum1;  
  vec_t accum2;
  vec_t accum3;  
  data_t result = (data_t)(0);
  pack_t xfer;

  /* initialize accum entries to 0 */
  for ( i = 0; i < VSIZE; i++) {
    xfer.d[i] = (data_t)(0);
  }
  accum0 = xfer.v;
  accum1 = xfer.v;  
  accum2 = xfer.v;
  accum3 = xfer.v;

  /* Single step until we have memory alignment */
  while ((((long) data0) % VBYTES) &&  cnt) {
    result += *data0++ * *data1++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism and dual accumulators */
  while (cnt >= 4*VSIZE) {
    vec_t v0chunk0 = *((vec_t *) data0);
    vec_t v1chunk0 = *((vec_t *) data1);
    vec_t v0chunk1 = *((vec_t *) (data0+VSIZE));
    vec_t v1chunk1 = *((vec_t *) (data1+VSIZE));
    vec_t v0chunk2 = *((vec_t *) (data0+2*VSIZE));
    vec_t v1chunk2 = *((vec_t *) (data1+2*VSIZE));
    vec_t v0chunk3 = *((vec_t *) (data0+3*VSIZE));
    vec_t v1chunk3 = *((vec_t *) (data1+3*VSIZE));
    accum0 = accum0 + (v0chunk0 * v1chunk0);
    accum1 = accum1 + (v0chunk1 * v1chunk1);
    accum2 = accum2 + (v0chunk2 * v1chunk2);
    accum3 = accum3 + (v0chunk3 * v1chunk3);
    data0 += 4*VSIZE;
    data1 += 4*VSIZE;
    cnt -= 4*VSIZE;
  }

  /* Single step through remaining elements */
  while (cnt) {
    result += *data0++ * *data1++;
    cnt--;
  }

  /* Combine elements of accumulator vector */
  xfer.v = accum0 + accum1 + accum2 + accum3;
  for ( i = 0; i < VSIZE; i++) {
    result += xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of dot8_4 */

/* dot8_8:  Vector, 8 accumulators */
void dot8_8(array_ptr v0, array_ptr v1, data_t *dest)
{
  long int i;
  long int get_array_length(array_ptr v);
  data_t *get_array_start(array_ptr v);
  long int cnt = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);
  vec_t accum0;
  vec_t accum1;  
  vec_t accum2;
  vec_t accum3;  
  vec_t accum4;
  vec_t accum5;  
  vec_t accum6;
  vec_t accum7;  
  data_t result = (data_t)(0);
  pack_t xfer;

  /* initialize accum entries to 0 */
  for ( i = 0; i < VSIZE; i++) {
    xfer.d[i] = (data_t)(0);
  }
  accum0 = xfer.v;
  accum1 = xfer.v;  
  accum2 = xfer.v;
  accum3 = xfer.v;
  accum4 = xfer.v;
  accum5 = xfer.v;  
  accum6 = xfer.v;
  accum7 = xfer.v;


  /* Single step until we have memory alignment */
  while ((((long) data0) % VBYTES) &&  cnt) {
    result += *data0++ * *data1++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism and dual accumulators */
  while (cnt >= 8*VSIZE) {
    vec_t v0chunk0 = *((vec_t *) data0);
    vec_t v1chunk0 = *((vec_t *) data1);
    vec_t v0chunk1 = *((vec_t *) (data0+VSIZE));
    vec_t v1chunk1 = *((vec_t *) (data1+VSIZE));
    vec_t v0chunk2 = *((vec_t *) (data0+2*VSIZE));
    vec_t v1chunk2 = *((vec_t *) (data1+2*VSIZE));
    vec_t v0chunk3 = *((vec_t *) (data0+3*VSIZE));
    vec_t v1chunk3 = *((vec_t *) (data1+3*VSIZE));
    vec_t v0chunk4 = *((vec_t *) (data0+4*VSIZE));
    vec_t v1chunk4 = *((vec_t *) (data1+4*VSIZE));
    vec_t v0chunk5 = *((vec_t *) (data0+5*VSIZE));
    vec_t v1chunk5 = *((vec_t *) (data1+5*VSIZE));
    vec_t v0chunk6 = *((vec_t *) (data0+6*VSIZE));
    vec_t v1chunk6 = *((vec_t *) (data1+6*VSIZE));
    vec_t v0chunk7 = *((vec_t *) (data0+7*VSIZE));
    vec_t v1chunk7 = *((vec_t *) (data1+7*VSIZE));
    accum0 = accum0 + (v0chunk0 * v1chunk0);
    accum1 = accum1 + (v0chunk1 * v1chunk1);
    accum2 = accum2 + (v0chunk2 * v1chunk2);
    accum3 = accum3 + (v0chunk3 * v1chunk3);
    accum4 = accum4 + (v0chunk4 * v1chunk4);
    accum5 = accum5 + (v0chunk5 * v1chunk5);
    accum6 = accum6 + (v0chunk6 * v1chunk6);
    accum7 = accum7 + (v0chunk7 * v1chunk7);
    data0 += 8*VSIZE;
    data1 += 8*VSIZE;
    cnt -= 8*VSIZE;
  }

  /* Single step through remaining elements */
  while (cnt) {
    result += *data0++ * *data1++;
    cnt--;
  }

  /* Combine elements of accumulator vector */
  xfer.v = accum0 + accum1 + accum2 + accum3 + accum4 + accum5 + accum6 + accum7;
  for ( i = 0; i < VSIZE; i++) {
    result += xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of dot8_8 */

