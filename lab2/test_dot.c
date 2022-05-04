/****************************************************************************

 gcc -O1 test_dot.c -lrt -o test_dot

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

/* We want to test a range of work sizes. We will generate these
   using the quadratic formula:  A x^2 + B x + C                     */
#define A   5   /* coefficient of x^2 */
#define B   100   /* coefficient of x */
#define C   50  /* constant term */

#define NUM_TESTS 20   /* Number of different sizes to test */


#define OUTER_LOOPS 2000

#define CPNS 2.4    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */

#define OPTIONS 4

/* Type of operation. This can be multiplication or addition.
   for addition, IDENT should be 0.0 and OP should be +
   for multiplication, IDENT should be 1.0 and OP should be *
 */
#define IDENT 1.0
#define OP +

/* Type of data being "combined". This can be any of the types:
   int, long int, float, double, long double */
typedef double data_t;

/* Create abstract data type for an array in memory */
typedef struct {
  long int len;
  data_t *data;
} array_rec, *array_ptr;

/* Prototypes */
array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
int set_array_length(array_ptr v, long int index);
int init_array(array_ptr v, long int len);
// void combine1(array_ptr v, data_t *dest);
// void combine2(array_ptr v, data_t *dest);
// void combine3(array_ptr v, data_t *dest);
// void combine4(array_ptr v, data_t *dest);
// void combine5(array_ptr v, data_t *dest);
// void combine6(array_ptr v, data_t *dest);
// void combine7(array_ptr v, data_t *dest);
void dot_prod(array_ptr v0,array_ptr v1,data_t *dest);
void dot_prod_unroll(array_ptr v0,array_ptr v1, data_t *dest);
void dot_prod_paral_multiple(array_ptr v0,array_ptr v1, data_t *dest);
void dot_prod_paral_reassociative(array_ptr v0,array_ptr v1, data_t *dest);



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
  double final_answer = 0;
  long int x, n, k, alloc_size;
  data_t *result;

  printf("Vector reduction (combine) examples\n");

  wakeup_delay();

  final_answer = wakeup_delay();

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  /* declare and initialize the arrays */
  array_ptr v0 = new_array(alloc_size);
  init_array(v0, alloc_size);
  array_ptr v1 = new_array(alloc_size);
  init_array(v1, alloc_size);
  result = (data_t *) malloc(sizeof(data_t));

  printf("Testing %d variants of combine(),\n", OPTIONS);
  printf("  on arrays of %d sizes from %d to %ld\n", NUM_TESTS, C, alloc_size);

  /* execute and time all 7 options from B&O  */
  OPTION = 0;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      dot_prod(v0,v1,result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      dot_prod_unroll(v0,v1,result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      dot_prod_paral_multiple(v0,v1,result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  OPTION++;
  printf("testing option %d\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    set_array_length(v0, n);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    for(k=0; k<OUTER_LOOPS; k++) {
      dot_prod_paral_reassociative(v0,v1,result);
      final_answer += *result;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
  }

  
  /* output times */
  printf("%s", "size, dot_prod, dot_prod_unroll, dot_prod_paral_multiple, dot_prod_paral_reassociative\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%d,  ", (A*i*i + B*i + C) * OUTER_LOOPS );
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) {
          printf(", ");
        }
        printf("%ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }

  printf("\n");
  printf("Sum of all results: %g\n", final_answer);

  return 0;  
} /* end main */

/**********************************************/
/* Create array of specified length */
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
  }
  else result->data = NULL;

  return result;
}

/* Retrieve array element and store at dest.
   Return 0 (out of bounds) or 1 (successful)
*/
int get_array_element(array_ptr v, long int index, data_t *dest)
{
  if (index < 0 || index >= v->len) {
    return 0;
  }
  *dest = v->data[index];
  return 1;
}

/* Return length of array */
long int get_array_length(array_ptr v)
{
  return v->len;
}

/* Set length of array */
int set_array_length(array_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize an array */
int init_array(array_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) {
      v->data[i] = (data_t)(i+1);
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

// model of combine4
void dot_prod(array_ptr v0,array_ptr v1,data_t *dest)
{
  long int i;
  long int length = get_array_length(v0);
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  data_t acc = IDENT;

  for (i = 0; i < length; i++) {
    acc = acc + data0[i] + data1[i];
  }
  *dest = acc;
}

// model of combine5
void dot_prod_unroll(array_ptr v0,array_ptr v1, data_t *dest)
{
  long int i;
  long int length = get_array_length(v0);
  long int limit = length - 1;
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=10) {
    acc = ((((((((((acc + data0[i] * data1[i]) + data0[i+1] * data1[i+1]) + data0[i+2] * data1[i+2]) + data0[i+3] * data1[i+3]) + data0[i+4] * data1[i+4]) + data0[i+5] * data1[i+5]) + data0[i+6] * data1[i+6]) + data0[i+7] * data1[i+7]) + data0[i+8] * data1[i+8]) + data0[i+9] * data1[i+9]);
    //acc = acc OP (data0[i] OP data1[i]) OP (data0[i+1] OP data1[i+1]) OP (data0[i+2] OP data1[i+2]);

    //acc = acc OP (data0[i] OP data1[i]) OP (data0[i+1] OP data1[i+1]) OP (data0[i+2] OP data1[i+2]) OP (data0[i+3] OP data1[i+3]) OP (data0[i+4] OP data1[i+4]) OP (data0[i+5] OP data1[i+5]) OP (data0[i+6] OP data1[i+6]) OP (data0[i+7] OP data1[i+7]);

    // acc = acc + (data0[i] * data1[i]) + (data0[i+1] * data1[i+1]) + (data0[i+2] * data1[i+2]) + (data0[i+3] * data1[i+3]) + (data0[i+4] * data1[i+4]) + (data0[i+5] * data1[i+5]) + (data0[i+6] * data1[i+6]) + (data0[i+7] * data1[i+7]) + (data0[i+8] * data1[i+8]) + (data0[i+9] * data1[i+9]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc + data0[i] * data1[i];
  }
  *dest = acc;
}

// model of combine6, multiple accumulators
void dot_prod_paral_multiple(array_ptr v0,array_ptr v1, data_t *dest)
{
  long int i;
  long int length = get_array_length(v0);
  long int limit = length - 1;
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  data_t acc0 = IDENT;
  data_t acc1 = IDENT;
  data_t acc2 = IDENT;
  data_t acc3 = IDENT;
  data_t acc4 = IDENT;
  data_t acc5 = IDENT;  
  data_t acc6 = IDENT;
  data_t acc7 = IDENT;
  data_t acc8 = IDENT;
  data_t acc9 = IDENT;


  /* Combine two elements at a time w/ 2 accumulators */
  for (i = 0; i < limit; i+=8) {
    acc0 = acc0 + data0[i] * data1[i];
    acc1 = acc1 + data0[i+1] * data1[i+1];
    acc2 = acc2 + data0[i+2] * data1[i+2];
    acc3 = acc3 + data0[i+3] * data1[i+3];
    acc4 = acc4 + data0[i+4] * data1[i+4];
    acc5 = acc5 + data0[i+5] * data1[i+5];
    acc6 = acc6 + data0[i+6] * data1[i+6];
    acc7 = acc7 + data0[i+7] * data1[i+7];
    acc8 = acc8 + data0[i+8] * data1[i+8];
    acc9 = acc9 + data0[i+9] * data1[i+9];

    // acc8 = acc8 OP data0[i+8] OP data1[i+8];
    // acc9 = acc9 OP data0[i+9] OP data1[i+9];

  }

  /* Finish remaining elements */
  for (; i < length; i++) {
   acc0 = acc0 + data0[i] * data1[i];

  }
  *dest = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9;

  //*dest = acc0 OP acc1 OP acc2 OP acc3 OP acc4 OP acc5 OP acc6 OP acc7 OP acc8 OP acc9;
}

// model of combine7
void dot_prod_paral_reassociative(array_ptr v0,array_ptr v1, data_t *dest)
{
  long int i;
  long int length = get_array_length(v0);
  long int limit = length - 1;
  data_t *data0 = get_array_start(v0);
  data_t *data1 = get_array_start(v1);

  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=10) {
    acc = acc + (data0[i] * data1[i]) + (data0[i+1] * data1[i+1]) + (data0[i+2] * data1[i+2]) + (data0[i+3] * data1[i+3]) + (data0[i+4] * data1[i+4]) + (data0[i+5] * data1[i+5]) + (data0[i+6] * data1[i+6]) + (data0[i+7] * data1[i+7]) + (data0[i+8] * data1[i+8]) + (data0[i+9] * data1[i+9]);
  }
  /* Finish remaining elements */
  for (; i < length; i++) {
    acc = acc + data0[i] * data1[i];
  }
  *dest = acc;
}

