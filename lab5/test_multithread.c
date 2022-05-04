/*
   gcc -O1 -std=gnu11 test_multithread.c -lpthread -lrt -lm -o test_multithread

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

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define GHOST 2   /* 2 extra rows/columns for "ghost zone". */

#define A   1200   /* coefficient of x^2 */
#define B   400  /* coefficient of x */
#define C   200  /* constant term */

#define NUM_TESTS 2

/* A, B, and C needs to be a multiple of your BLOCK_SIZE,
   total array size will be (GHOST + Ax^2 + Bx + C) */

#define BLOCK_SIZE 32     // TO BE DETERMINED

#define OPTIONS 3

#define MINVAL   0.0
#define MAXVAL  10.0

#define TOL 0.00001
#define OMEGA 1.90       // TO BE DETERMINED

#define NUM_THREADS 4

typedef double data_t;

typedef struct {
  long int rowlen;
  data_t *data;
} arr_rec, *arr_ptr;

struct thread_data{
  long thread_id;
  arr_ptr v;
};

/* Barrier variable */
pthread_barrier_t barrier1;

double total_change = 1.0e10;
pthread_mutex_t mutexA;   /* declare a global mutex */
int iterations_thread = 0;
pthread_mutex_t mutexB;   /* declare a global mutex */


/* Prototypes */
arr_ptr new_array(long int row_len);
int set_arr_rowlen(arr_ptr v, long int index);
long int get_arr_rowlen(arr_ptr v);
int init_array(arr_ptr v, long int row_len);
int init_array_rand(arr_ptr v, long int row_len);
int print_array(arr_ptr v);

void SOR(arr_ptr v, int *iterations);
void *SOR_threaded_adj_rows(void * threadarg);
void *SOR_threaded_nonadj_rows(void * threadarg);

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

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  int convergence[OPTIONS][NUM_TESTS];
  int *iterations;
  struct thread_data thread_data_array[NUM_THREADS];

  long int x, n;
  long int alloc_size;
  int rc;

  pthread_t threads[NUM_THREADS];

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
  printf("OPTION=%d (SOR_threaded_adj_rows)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    total_change = 1.0e10;
    iterations_thread = 0;

    /* Barrier initialization -- spawned threads will wait until all arrive */
    if (pthread_barrier_init(&barrier1, NULL, NUM_THREADS)) {
      printf("Could not create a barrier\n");
      return -1;
    } 
    clock_gettime(CLOCK_REALTIME, &time_start);


    for (long t = 0; t < NUM_THREADS; t++) {
      thread_data_array[t].thread_id = t;
      thread_data_array[t].v = v0;
      rc = pthread_create(&threads[t], NULL, SOR_threaded_adj_rows,
                                             (void*) &thread_data_array[t]);
      if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }

    for (long t = 0; t < NUM_THREADS; t++) {
      if (pthread_join(threads[t], NULL)) {
        printf("ERROR; pthread_join() failed");
        exit(19);
      }
    }

    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = iterations_thread;
  }

  OPTION++;
  printf("OPTION=%d (SOR_threaded_nonadj_rows)\n", OPTION);
  for (x=0; x<NUM_TESTS && (n = A*x*x + B*x + C, n<=alloc_size); x++) {
    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    total_change = 1.0e10;
    iterations_thread = 0;
    /* Barrier initialization -- spawned threads will wait until all arrive */
    if (pthread_barrier_init(&barrier1, NULL, NUM_THREADS)) {
      printf("Could not create a barrier\n");
      return -1;
    } 
    clock_gettime(CLOCK_REALTIME, &time_start);  

    for (long t = 0; t < NUM_THREADS; t++) {
      thread_data_array[t].thread_id = t;
      thread_data_array[t].v = v0;
      rc = pthread_create(&threads[t], NULL, SOR_threaded_nonadj_rows,
                                             (void*) &thread_data_array[t]);
      if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }

    for (long t = 0; t < NUM_THREADS; t++) {
      if (pthread_join(threads[t], NULL)) {
        exit(19);
      }
    }

    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    convergence[OPTION][x] = iterations_thread;
  }

  printf("All times are in cycles (if CPNS is set correctly in code)\n");
  printf("\n");
  printf("size, SOR time, SOR iters, SOR_threaded_adj_rows time, SOR_threaded_adj_rows iters, SOR_threaded_nonadj_rows time, SOR_threaded_nonadj_rows iters\n");
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
     with stripes, red/black, ...), it is more useful to have the same
     randomness for any given array size */
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

void *SOR_threaded_adj_rows(void * threadarg)
{
    struct thread_data *my_data;
    long thread_id;
    arr_ptr v;
    my_data = (struct thread_data *) threadarg;
    thread_id = my_data->thread_id;
    v = my_data->v;

    long int i, j;
    long int rowlen = get_arr_rowlen(v);
    data_t *data = get_array_start(v);
    double change, mychange;   /* start w/ something big */
    int rc;
    int done = 0;

    int mymin = 1 + (thread_id * rowlen / NUM_THREADS);
    int mymax = mymin + rowlen / NUM_THREADS - 1;

    while (!done) {
        if (pthread_mutex_lock(&mutexB)) {
          printf("ERROR on lock\n");
        }
        iterations_thread++;
        if (pthread_mutex_unlock(&mutexB)) {  /* unlock thread */
          printf("ERROR on unlock\n");
        }

        mychange = total_change = 0;
        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
          exit(-1);
        }
        for (i = mymin; i < mymax; i++) {
          for (j = 1; j < rowlen-1; j++) {
              change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                              data[(i+1)*rowlen+j] +
                                              data[i*rowlen+j+1] +
                                              data[i*rowlen+j-1]);
              data[i*rowlen+j] -= change * OMEGA;
              if (change < 0){
              change = -change;
              }

              mychange += change;
          }
        }
        if (pthread_mutex_lock(&mutexA)) {
          printf("ERROR on lock\n");
        }
        total_change += mychange;
        if (pthread_mutex_unlock(&mutexA)) {  /* unlock thread */
          printf("ERROR on unlock\n");
        }

        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
          exit(-1);
        }
        if ((total_change/(double)(rowlen*rowlen)) <= (double)TOL) done = 1;
        if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
          printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iterations_thread);
          break;
        }
        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
          exit(-1);
        }
    }
    printf("    thread_id = %ld, SOR_threaded_adj_rows() done after totally %d iters\n", thread_id, iterations_thread);
}

void *SOR_threaded_nonadj_rows(void * threadarg)
{
    struct thread_data *my_data;
    long thread_id;
    arr_ptr v;
    my_data = (struct thread_data *) threadarg;
    thread_id = my_data->thread_id;
    v = my_data->v;

    long int i, j;
    long int rowlen = get_arr_rowlen(v);
    data_t *data = get_array_start(v);
    double change, mychange;   /* start w/ something big */
    int rc;
    int done = 0;

    int mymin = 1 + thread_id;
    int step = NUM_THREADS;
    // int mymax = mymin + rowlen / NUM_THREADS - 1;

    while (!done) {
        if (pthread_mutex_lock(&mutexB)) {
          printf("ERROR on lock\n");
        }
        iterations_thread++;
        if (pthread_mutex_unlock(&mutexB)) {  /* unlock thread */
          printf("ERROR on unlock\n");
        }

        mychange = total_change = 0;
        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
          exit(-1);
        }
        for (i = mymin; i < rowlen-1; i+=step) {
          for (j = 1; j < rowlen-1; j++) {
              change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                              data[(i+1)*rowlen+j] +
                                              data[i*rowlen+j+1] +
                                              data[i*rowlen+j-1]);
              data[i*rowlen+j] -= change * OMEGA;
              if (change < 0){
              change = -change;
              }

              mychange += change;
          }
        }
        if (pthread_mutex_lock(&mutexA)) {
          printf("ERROR on lock\n");
        }
        total_change += mychange;
        if (pthread_mutex_unlock(&mutexA)) {  /* unlock thread */
          printf("ERROR on unlock\n");
        }

        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
          exit(-1);
        }
        if ((total_change/(double)(rowlen*rowlen)) <= (double)TOL) done = 1;
        if (abs(data[(rowlen-2)*(rowlen-2)]) > 10.0*(MAXVAL - MINVAL)) {
          printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iterations_thread);
          break;
        }
        rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
          exit(-1);
        }
    }
    printf("   thread_id = %ld, SOR_threaded_nonadj_rows() done after totally %d iters\n", thread_id, iterations_thread);
}