/*************************************************************************/
// gcc -pthread test_sor.c -o test_sor -std=gnu99

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>


#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
# include "apple_pthread_barrier.h"
#endif // __APPLE__

/* Imagine a long metal bar that is initially at room temperature.
   Heating and cooling devices are applied to the ends of the bar so that
   one end is kept at a constant temperature of 0 degrees, the other end
   at 100 degrees. 

   We can simulate this system with N threads, each one responsible for
   tracking the temperature at a point along the bar, with successive
   calculations representing successive points in time.

   We'll be calculating an approximate solution of the heat equation on
   a 1-dimensional grid of N+2 points. There are 2 extra points to hold
   the boundary values, which don't change.

   If N=10, the array will have 12 elements. If the boundary conditions
   (temperatures at the ends) are 0 and 100, we have:

   +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
   |  0  | t_0 | t_1 | t_2 | t_3 | t_4 | t_5 | t_6 | t_7 | t_8 | t_9 | 100 |
   +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

   where t_i are values that vary with time.

   Given the two endpoints as boundary conditions in space, and any initial
   set of values t_i, using the heat equation or diffusion equation, there is
   one solution in the limit as time goes to infinity (i.e. a steady-state
   equilibrium):

    0.00  9.09 18.18 27.27 36.36 45.45 54.55 63.64 72.73 81.82 90.91 100.00

*/

#define USE_BARRIERS 1

#define ITERS 14000
#define DELTA_T 0.01
#define DELTA_X 1.0

#define NUM_THREADS 10

double heatvals[NUM_THREADS+2];
double hvals_2[NUM_THREADS+2];

struct thread_data {
  int thread_id;
  int array_index;
};

pthread_barrier_t barrier1;

/* Worker thread for Successive Over-relaxation */
void *worker(void *threadarg)
{
  struct thread_data *my_data;
  int tid;
  long i;
  double t;
  int rc;
  double * hv_from;
  double * hv_to;

  my_data = (struct thread_data *) threadarg;
  tid = my_data->thread_id;
  i = my_data->array_index;

  printf("Thread for index %ld starting\n", i);

  for (long iter=0; iter<ITERS; iter++) {
    /* We read from one array and write to another. This is a common
       technique sometimes called "ping-pong". */
    if (iter % 2) {
      hv_from = heatvals; hv_to = hvals_2;
    } else {
      hv_from = hvals_2; hv_to = heatvals;
    }

    /* Forward Euler integration technique for the heat equation on
       a one-dimensional grid.

       U is a 1-D array U of temperatures

       U_i is the current temperature of the ith element at time t
       U_i' is the temperature at time t+DELTA_T

       Using a forward difference in time and a central difference in space,
       the solution to the heat equation can be approximated by

       (U_i' - U_i)/DELTA_T = (U_(i-1) - 2 U_i + U_(i+1)) / DELTA_X^2

       (adapted from http://hplgit.github.io/num-methods-for-PDEs/doc/pub/diffu/sphinx/._main_diffu001.html )

       This calculation is similar to the SOR ("Successive Over-Relaxation")
       technique which finds the steady-state equilibrium solution for a
       system that has one.  However, this calculation is more general-purpose
       in that it can also simulate systems that have no steady-state solution.
     */

    t = hv_from[i-1] - 2.0*hv_from[i] + hv_from[i+1];
    hv_to[i] = hv_from[i] + DELTA_T * t / (DELTA_X*DELTA_X);

    if (USE_BARRIERS) {
      rc = pthread_barrier_wait(&barrier1);
      if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        printf("Thread %d failed on barrier wait\n", tid);
        exit(-1);
      }
    }
  }

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  clock_t begin = clock();
  int i;
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  long t;

  printf("test_sor setup...\n");

  /* Initialise the array of heat values */
  for (long t=0; t<NUM_THREADS+2; t++) {
    heatvals[t] = 25.0;
  }
  heatvals[0] = 0.0;
  heatvals[NUM_THREADS+1] = 100.0;
  /* Copy the pattern into the second array */
  for (long t=0; t<NUM_THREADS+2; t++) {
    hvals_2[t] = heatvals[t];
  }

  /* -------------------- Add Code Here -------------------- */

  if (pthread_barrier_init(&barrier1, NULL, NUM_THREADS)) {
    printf("Could not create a barrier\n");
    return -1;
  } 
 

  for (t = 0; t < NUM_THREADS; t++){
    thread_data_array[t].thread_id = t;
    thread_data_array[t].array_index = t+1;
    rc = pthread_create(&threads[t], NULL, worker, (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  printf("\nsystem state after %d iterations:\n", ITERS);

  for (t = 0; t < NUM_THREADS ; t++) {
    if (pthread_join(threads[t],NULL)){
      printf("\n ERROR on join\n");
      exit(19);
    }
  }

  /* Display the result */
  for (long t=0; t<NUM_THREADS+2; t++) {
    printf(" %5.2f", heatvals[t]);
  }
  printf("\n");


  clock_t end = clock();
  double time_taken = (double)(end-begin)/CLOCKS_PER_SEC; // in seconds 
  printf("worker() took %f seconds to execute \n", time_taken); 

  pthread_exit(NULL);


} /* end main */