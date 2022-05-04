/*****************************************************************************

 gcc -O1 -std=gnu99 -mavx test_combine8.c -lrt -lm -o test_combine8

Vector reduction functions:
 
  combine4   -- base scalar code
  combine6_5 -- unrolled 5 times with 5 accumulators -- "best" scalar code
  combine8   -- base vector code
  combine8_2 -- unrolled 2 times with 2 accumulators  NEED TO ADD
  combine8_4 -- unrolled 4 times with 4 accumulators
  combine8_8 -- unrolled 8 times with 8 accumulators  NEED TO ADD

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

/* When you add two more combine functions, change OPTIONS to 6 */
#define OPTIONS 6

/* Modify to select add (IDENT=0.0, OP=+) or multiply (IDENT=1.0, OP=*) */
#define IDENT 0.0
#define OP +

typedef double data_t;

/* Create abstract data type for an array in memory */
typedef struct {
  long int len;
  data_t *data;
} array_rec, *array_ptr;

/* Number of bytes in a vector */
#define VBYTES 32

/* Number of elements in a vector */
#define VSIZE (VBYTES/sizeof(data_t))

// define the vector data type 
/* This declaration states that data type vec_t is vector, where the elements are of type data_t, and the
vector consists of VBYTES bytes */
typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));

// use union declaration to overlay vector and array data types
typedef union {
  vec_t v;
  data_t d[VSIZE];
} pack_t;

array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
int set_array_length(array_ptr v, long int index);
int init_array(array_ptr v, long int len);

void combine4(array_ptr v, data_t *dest);
void combine6_5(array_ptr v, data_t *dest);
void combine8(array_ptr v, data_t *dest);
void combine8_2(array_ptr v, data_t *dest);
void combine8_4(array_ptr v, data_t *dest);
void combine8_8(array_ptr v, data_t *dest);



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

  /* This unusual variable declaration causes an error if the program is
     compiled on older machines that do not have AVX capability. For
     quite a few years, students were trying to use their own computers
     to compile this program and getting odd error messages. */
  __m256 PLEASE_USE_LAB_MACHINES_FOR_THIS_ASSIGNMENT;

  printf("reduction -- vector examples\n");

  wd = wakeup_delay();
  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  /* declare and initialize the vector structure */
  array_ptr v0 = new_array(alloc_size);
  data_holder = (data_t *) malloc(sizeof(data_t));
  init_array(v0, alloc_size);

  /* execute and time ... options from B&O  */
  OPTION = 0;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      combine4(v0, data_holder);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      combine6_5(v0, data_holder);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      combine8(v0, data_holder);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      combine8_2(v0, data_holder);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      combine8_4(v0, data_holder);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for (long k=0; k<OUTER_LOOPS; k++) {
      combine8_8(v0, data_holder);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  /* When you add combine8_2 and combine8_8 don't forget to change this printf too */
  printf("size, combine4, combine6_5, combine8, combine8_2, combine8_4, combine8_8 \n");
  {
    for (int i = 0; i < x; i++) {
      printf("%ld,  ", (long)((A*i*i + B*i + C) * OUTER_LOOPS) );
      for (int j = 0; j < OPTIONS; j++) {
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
    data_t *data = (data_t *) calloc(len + VSIZE, sizeof(data_t));
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

/* Set length of an array. This does NOT change the amount of memory
   allocated for the array, it only changes the field that controls how
   many elements are used when doing one of the computation operations. */
int set_array_length(array_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize an array to consecutive integers (converted to data_t type) */
int init_array(array_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) {
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

/*************************************************/
/* Combine4:  Accumulate result in local variable
 * Example of --> Eliminate unneeded memory references */
void combine4(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  data_t *data = get_array_start(v);
  data_t acc = IDENT;

  for (i = 0; i < length; i++) {
    acc = acc OP data[i];
  }
  *dest = acc;
} /* End of combine4 */

/* Combine6_5:  Unroll loop by 5, using 5 accumulators
 * Example of --> parallelism */
void combine6_5(array_ptr v, data_t *dest)
{
  long int i;
  long int length = get_array_length(v);
  long int limit = length - 1;
  data_t *data = get_array_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;
  data_t acc3 = IDENT;
  data_t acc4 = IDENT;

  /* Combine 5 elements at a time w/ 5 acculators */
  for (i = 0; i < limit; i+=5) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
    acc2 = acc2 OP data[i+2];
    acc3 = acc3 OP data[i+3];
    acc4 = acc4 OP data[i+4];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 = acc0 OP data[i];
  }
  *dest = acc0 OP acc1 OP acc2 OP acc3 OP acc4;
} /* End of combine6_5 */

/* Combine8:  Vector version */
void combine8(array_ptr v, data_t *dest)
{
  long int i;
  pack_t xfer;
  vec_t accum;
  data_t *data = get_array_start(v);
  long int cnt = get_array_length(v);
  data_t result = IDENT;

  /* Initialize accum entries to IDENT  初始化xfer的d值*/ 
  // VSIZE =8 VBYTES=32
  for (i = 0; i < VSIZE; i++) {
    xfer.d[i] = IDENT;
  }
  accum = xfer.v;

  /* Single step until we have memory alignment  在内存对齐之前都进行单步相加*/
  while (((long) data) % VBYTES && cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism 用VSIZE并行的方式处理数据*/
  while (cnt >= VSIZE) {
    vec_t chunk = *((vec_t *) data);
    accum = accum OP chunk;
    data += VSIZE;
    cnt -= VSIZE;
  }

  /* Single-step through the remaining elements 处理剩余数据*/
  while (cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Combine elements of accumulator vector 合并累加器向量的元素*/
  xfer.v = accum;
  for (i = 0; i < VSIZE; i++) {
    result = result OP xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of combine8 */



/* You will add a combine8_2 here */
/* Combine8_2:  Vector 2x multiple accumulators */
void combine8_2(array_ptr v, data_t *dest)
{
  long int i;
  pack_t xfer;
  vec_t accum0;
  vec_t accum1;
  data_t *data = get_array_start(v);
  long int cnt = get_array_length(v);
  data_t result = IDENT;

  /* Initialize accum entries to IDENT */
  for (i = 0; i < VSIZE; i++) {
    xfer.d[i] = IDENT;
  }
  accum0 = xfer.v;
  accum1 = xfer.v;

  /* Single step until we have memory alignment */
  while (((long) data) % VBYTES && cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism */
  while (cnt >= 2*VSIZE) {
    vec_t chunk0 = *((vec_t *) data);
    vec_t chunk1 = *((vec_t *) data+VSIZE);
    accum0 = accum0 OP chunk0;
    accum1 = accum1 OP chunk1;
    data += 2*VSIZE;
    cnt -= 2*VSIZE;
  }

  /* Single-step through the remaining elements */
  while (cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Combine elements of accumulator vectors into a single vector */
  xfer.v = accum0 OP accum1;
  /* Combine elements of this single vector into a single value */
  for (i = 0; i < VSIZE; i++) {
    result = result OP xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of combine8_2 */


/* Combine8_4:  Vector 4x multiple accumulators */
void combine8_4(array_ptr v, data_t *dest)
{
  long int i;
  pack_t xfer;
  vec_t accum0;
  vec_t accum1;
  vec_t accum2;
  vec_t accum3;
  data_t *data = get_array_start(v);
  long int cnt = get_array_length(v);
  data_t result = IDENT;

  /* Initialize accum entries to IDENT */
  for (i = 0; i < VSIZE; i++) {
    xfer.d[i] = IDENT;
  }
  
  accum0 = xfer.v;
  accum1 = xfer.v;
  accum2 = xfer.v;
  accum3 = xfer.v;

  /* Single step until we have memory alignment */
  while (((long) data) % VBYTES && cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism */
  while (cnt >= 4*VSIZE) {
    vec_t chunk0 = *((vec_t *) data);
    vec_t chunk1 = *((vec_t *) data+VSIZE);
    vec_t chunk2 = *((vec_t *) data+2*VSIZE);
    vec_t chunk3 = *((vec_t *) data+3*VSIZE);
    accum0 = accum0 OP chunk0;
    accum1 = accum1 OP chunk1;
    accum2 = accum2 OP chunk2;
    accum3 = accum3 OP chunk3;
    data += 4*VSIZE;
    cnt -= 4*VSIZE;
  }

  /* Single-step through the remaining elements */
  while (cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Combine elements of accumulator vectors into a single vector */
  xfer.v = accum0 OP accum1 OP accum2 OP accum3;

  /* Combine elements of this single vector into a single value */
  for (i = 0; i < VSIZE; i++) {
    result = result OP xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of combine8_4 */


/* You will add a combine8_8 here */
void combine8_8(array_ptr v, data_t *dest)
{
  long int i;
  pack_t xfer;
  vec_t accum0;
  vec_t accum1;
  vec_t accum2;
  vec_t accum3;
  vec_t accum4;
  vec_t accum5;
  vec_t accum6;
  vec_t accum7;
  data_t *data = get_array_start(v);
  long int cnt = get_array_length(v);
  data_t result = IDENT;

  /* Initialize accum entries to IDENT */
  for (i = 0; i < VSIZE; i++) {
    xfer.d[i] = IDENT;
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
  while (((long) data) % VBYTES && cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Step through data with VSIZE-way parallelism */
  while (cnt >= 8*VSIZE) {
    vec_t chunk0 = *((vec_t *) data);
    vec_t chunk1 = *((vec_t *) data+VSIZE);
    vec_t chunk2 = *((vec_t *) data+2*VSIZE);
    vec_t chunk3 = *((vec_t *) data+3*VSIZE);
    vec_t chunk4 = *((vec_t *) data+4*VSIZE);
    vec_t chunk5 = *((vec_t *) data+5*VSIZE);
    vec_t chunk6 = *((vec_t *) data+6*VSIZE);
    vec_t chunk7 = *((vec_t *) data+7*VSIZE);

    accum0 = accum0 OP chunk0;
    accum1 = accum1 OP chunk1;
    accum2 = accum2 OP chunk2;
    accum3 = accum3 OP chunk3;
    accum4 = accum4 OP chunk4;
    accum5 = accum5 OP chunk5;
    accum6 = accum6 OP chunk6;
    accum7 = accum7 OP chunk7;
    data += 8*VSIZE;
    cnt -= 8*VSIZE;
  }

  /* Single-step through the remaining elements */
  while (cnt) {
    result = result OP *data++;
    cnt--;
  }

  /* Combine elements of accumulator vectors into a single vector */
  xfer.v = accum0 OP accum1 OP accum2 OP accum3 OP accum4 OP accum5 OP accum6 OP accum7;

  /* Combine elements of this single vector into a single value */
  for (i = 0; i < VSIZE; i++) {
    result = result OP xfer.d[i];
  }

  /* store result */
  *dest = result;
} /* End of combine8_8 */
