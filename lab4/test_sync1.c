/*************************************************************************

  gcc -pthread test_sync1.c -o test_sync -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 2    /* NEED TO ADJUST Messages MANUALLY */

struct thread_data{
  int thread_id;
  int sum;
  char *message;
};

pthread_mutex_t mutexA;   /* declare a global mutex */

/*************************************************************************/
void *PrintHello(void *threadarg)
{
  long taskid, sum;
  struct thread_data *my_data;
  char *message;

  my_data = (struct thread_data *) threadarg;
  taskid = my_data->thread_id;
  sum = my_data->sum;
  message = my_data->message;

  while (pthread_mutex_trylock(&mutexA));  /* wait until released */
  
  printf("in thread ID %lx (sum = %ld message = %s), now unblocked!\n",
        taskid, sum, message);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  long t;
  char *Messages[NUM_THREADS] = {"First Message" ,     //Adjust manually
                                 "Second Message"/*
                                 "Third Message",
                                 "Fourth Message",
                                 "Fifth Message" */ };
  char dummy[10];

  if (pthread_mutex_init(&mutexA, NULL)) {
    printf("ERROR initializing the lock\n");
    exit(-1);
  }

  printf("Hello sync1\n");

  for (t = 0; t < NUM_THREADS; t++) {
    if (pthread_mutex_lock(&mutexA)) {
      printf("ERROR on lock\n");
    }
    thread_data_array[t].thread_id = t;
    thread_data_array[t].sum = t+123;
    thread_data_array[t].message = Messages[t];
    rc = pthread_create(&threads[t], NULL, PrintHello,
                                      (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  printf("In main: created thread %ld, which is blocked\n", t);
  printf("   Press any letter key (not space) then press enter:\n");
  scanf("%s", dummy);
  if (pthread_mutex_unlock(&mutexA)) {  /* unlock thread */
    printf("ERROR on unlock\n");
  }

  for (t = 0; t < NUM_THREADS; t++) {
    if (pthread_join(threads[t], NULL)){
      printf(" ERROR on join\n");
      scanf("%s", dummy);
      exit(19);
    }
  }
  printf("After joining\n");

  printf("GOODBYE WORLD! \n");
  return(0);
} /* end main */