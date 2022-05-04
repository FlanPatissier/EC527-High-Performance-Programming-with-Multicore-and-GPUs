/************************************************************************

  gcc -pthread test_pt.c -lpthread -lm -lrt -o test_pt

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#define CPNS 2.4    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define A   1   /* coefficient of x^2 */
#define B   1   /* coefficient of x */
#define C   10  /* constant term */

#define NUM_TESTS 10

#define OPTIONS 4        // Current setting, vary as you wish!
#define IDENT 0

#define INIT_LOW -10.0
#define INIT_HIGH 10.0

typedef double data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int rowlen;
  data_t *data;
} matrix_rec, *matrix_ptr;

int NUM_THREADS = 4;

/* used to pass parameters to worker threads */
struct thread_data{
  int thread_id;
  matrix_ptr a;
  matrix_ptr b;
  matrix_ptr c;
  matrix_ptr d;
};

/* prototypes */
matrix_ptr new_matrix(long int row_len);
int set_matrix_rowlen(matrix_ptr m, long int index);
long int get_matrix_rowlen(matrix_ptr m);
int init_matrix(matrix_ptr m, long int row_len);
int init_matrix_rand(matrix_ptr m, long int row_len);
int init_matrix_rand_grad(matrix_ptr m, long int row_len);
int zero_matrix(matrix_ptr m, long int len);
void pt_cb_base(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void pt_cb_pthr(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void pt_mb(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d);
void pt_ob(matrix_ptr a, matrix_ptr b, matrix_ptr c);

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
      clock_gettime(CLOCK_REALTIME, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_REALTIME, &time_stop);
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

/*************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double wd;
  long int x, n;
  long int alloc_size;

  x = NUM_TESTS-1;
  alloc_size = A*x*x + B*x + C;

  printf("Test SOR pthreads\n");
  wd = wakeup_delay();

  /* declare and initialize the matrix structure */
  matrix_ptr a0 = new_matrix(alloc_size);
  init_matrix_rand(a0, alloc_size);
  matrix_ptr b0 = new_matrix(alloc_size);
  init_matrix_rand(b0, alloc_size);
  matrix_ptr c0 = new_matrix(alloc_size);
  zero_matrix(c0, alloc_size);
  matrix_ptr d0 = new_matrix(alloc_size);
  init_matrix_rand_grad(d0, alloc_size);

  OPTION = 0;
  printf("OPTION %d - pt_cb_base()\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    init_matrix_rand(a0, n);
    set_matrix_rowlen(a0, n);
    set_matrix_rowlen(b0, n);
    set_matrix_rowlen(c0, n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    pt_cb_base(a0, b0, c0);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    printf("iter %d done\n", x);
  }

  NUM_THREADS = 2;
  OPTION++;
  printf("OPTION %d: pt_cb_pthr() with %d threads\n", OPTION, NUM_THREADS);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    init_matrix_rand(a0, n);
    set_matrix_rowlen(a0, n);
    set_matrix_rowlen(b0, n);
    set_matrix_rowlen(c0, n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    pt_cb_pthr(a0, b0, c0);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    printf("iter %d done\n", x);
  }

  NUM_THREADS = 4;
  OPTION++;
  printf("OPTION %d: pt_cb_pthr() with %d threads\n", OPTION, NUM_THREADS);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    init_matrix_rand(a0, n);
    set_matrix_rowlen(a0, n);
    set_matrix_rowlen(b0, n);
    set_matrix_rowlen(c0, n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    pt_cb_pthr(a0, b0, c0);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    printf("iter %d done\n", x);
  }

  /* enable this to try the experiment on a machine with 8+ cores, and don't
     forget to also change OPTIONS definition at top! */
  
  NUM_THREADS = 8;
  OPTION++;
  printf("OPTION %d: pt_cb_pthr() with %d threads\n", OPTION, NUM_THREADS);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    init_matrix_rand(a0, n);
    set_matrix_rowlen(a0, n);
    set_matrix_rowlen(b0, n);
    set_matrix_rowlen(c0, n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    pt_cb_pthr(a0, b0, c0);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    printf("iter %d done\n", x);
  }
  

  printf("\n");
  printf("All measurements are in cycles (if CPNS is set correctly in the code)\n");
  printf("row length, 1 thread, 2 threads, 4 threads, 8 threads\n");
  {
    int i, j;
    for (i = 0; i < NUM_TESTS; i++) {
      printf("%d, ", A*i*i + B*i + C);
      for (j = 0; j < OPTIONS; j++) {
        if (j != 0) printf(", ");
        printf("%ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      }
      printf("\n");
    }
  }

  printf("test_pt done\n");
  printf("Wakeup delay calculated %f\n", wd);

  return 0;
} /* end main */

/**********************************************/
/* Create matrix of specified row length */
matrix_ptr new_matrix(long int row_len)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
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
                                            row_len*row_len*sizeof(data_t));
	  return NULL;  /* Couldn't allocate storage */
	}
	result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, long int row_len)
{
  m->rowlen = row_len;
  return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m)
{
  return m->rowlen;
}

/* initialize matrix to consecutive numbers starting with 0 */
int init_matrix(matrix_ptr m, long int row_len)
{
  long int i;

  if (row_len > 0) {
    m->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

/* initialize matrix to random values in [INIT_LOW, INIT_HIGH] */
int init_matrix_rand(matrix_ptr m, long int row_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  if (row_len > 0) {
    m->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++)
      m->data[i] = (data_t)(fRand((double)(INIT_LOW),(double)(INIT_HIGH)));
    return 1;
  }
  else return 0;
}

/* initialize matrix to random values bounded by a "gradient" envelope */
int init_matrix_rand_grad(matrix_ptr m, long int row_len)
{
  long int i;
  double fRand(double fMin, double fMax);

  if (row_len > 0) {
    m->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++)
      m->data[i] = (data_t)(fRand((double)(0.0),(double)(i)));
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int row_len)
{
  long int i,j;

  if (row_len > 0) {
    m->rowlen = row_len;
    for (i = 0; i < row_len*row_len; i++)
      m->data[i] = (data_t)(IDENT);
    return 1;
  }
  else return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

/* print matrix */
int print_matrix(matrix_ptr v)
{
  long int i, j, row_len;

  row_len = v->rowlen;
  for (i = 0; i < row_len; i++) {
    printf("\n");
    for (j = 0; j < row_len; j++)
      printf("%.4f ", (data_t)(v->data[i*row_len+j]));
  }
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / (double)(RAND_MAX);
  return fMin + f * (fMax - fMin);
}

/*************************************************/
/* CPU bound baseline - perform transcendental function on all elements */
void pt_cb_base(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int rowlen = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < rowlen*rowlen; i++) {
    c0[i] = (data_t)(cosh(tan(sqrt(cos(exp((double)(a0[i])))))));
    // c0[i] = a0[i];
  }
}

/***************************************************************************/
/* CPU bound multithreaded code. Here we use pthreads to do the same thing */
/* as pt_cb_base().  first, the worker thread function                     */
void *cb_work(void *threadarg)
{
  long int i, j, k, low, high;
  struct thread_data *my_data;
  my_data = (struct thread_data *) threadarg;
  int taskid = my_data->thread_id;
  matrix_ptr a0 = my_data->a;
  matrix_ptr b0 = my_data->b;
  matrix_ptr c0 = my_data->c;
  long int rowlen = get_matrix_rowlen(a0);
  data_t *aM = get_matrix_start(a0);
  data_t *bM = get_matrix_start(b0);
  data_t *cM = get_matrix_start(c0);

  low = (taskid * rowlen * rowlen)/NUM_THREADS;
  high = ((taskid+1)* rowlen * rowlen)/NUM_THREADS;

  for (i = low; i < high; i++) {
    cM[i] = (data_t)(cosh(tan(sqrt(cos(exp((double)(aM[i])))))));
    // cM[i] = aM[i];
  }

  pthread_exit(NULL);
} /* End of cb_work */

/* Now, the pthread calling function */
void pt_cb_pthr(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  long t;

  for (t = 0; t < NUM_THREADS; t++) {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].a = a;
    thread_data_array[t].b = b;
    thread_data_array[t].c = c;
    thread_data_array[t].d = 0;
    rc = pthread_create(&threads[t], NULL, cb_work,
			(void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  for (t = 0; t < NUM_THREADS; t++) {
    if (pthread_join(threads[t],NULL)){ 
      printf("ERROR; code on return from join is %d\n", rc);
      exit(-1);
    }
  }
}