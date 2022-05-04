/************************************************************************

   gcc -pthread test_barrier.c -o test_barrier -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
# include "apple_pthread_barrier.h"
#endif // __APPLE__

#define NUM_THREADS 5

struct thread_data{
  int thread_id;
  int sum;
  char *message;
};

/* Barrier variable */
pthread_barrier_t barrier1;

/********************/
void *PrintHello(void *threadarg)
{
  long taskid, sum;
  struct thread_data *my_data;
  char *message;
  int rc;

  my_data = (struct thread_data *) threadarg;
  taskid = my_data->thread_id;
  sum = my_data->sum;
  message = my_data->message;

  /* print some thread-specific stuff, then wait for everybody else */
  for (long i = 0; i < 3; i++) {
    sleep(i+2); 
    printf("Thread %ld printing before barrier %ld of 3\n", taskid, i+1);
  }

  /* wait at barrier until all NUM_THREAD threads arrive */
  rc = pthread_barrier_wait(&barrier1);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier (return code %d)\n", rc);
    exit(-1);
  }

  /* now print some more thread-specific stuff */
  for (long i = 0; i < 2; i++) {
    printf("Thread %ld after barrier (print %ld of 2)\n", taskid, i+1);
  }

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  int arg, i, j, k, m;
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  char *Messages[NUM_THREADS] = {"First Message",
				 "Second Message",
				 "Third Message",
				 "Fourth Message",
				 "Fifth Message"};
  char dummy[10];

  printf("Hello test_barrier.c\n");

  /* Barrier initialization -- spawned threads will wait until all arrive */
  if (pthread_barrier_init(&barrier1, NULL, NUM_THREADS)) {
    printf("Could not create a barrier\n");
    return -1;
  } 
 
  /* create threads */
  for (long t = 0; t < NUM_THREADS; t++) {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].sum = t+28;
    thread_data_array[t].message = Messages[t];
    printf("In main:  creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, PrintHello,
                                             (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  printf("After creating the threads; my id is %lx\n", pthread_self());

  for (long t = 0; t < NUM_THREADS; t++) {
    if (pthread_join(threads[t],NULL)){
      printf(" ERROR on join\n");
      exit(19);
    }
  }
  printf("After joining\n");

  return(0);

} /* end main */
