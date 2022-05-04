/*************************************************************************

  gcc -pthread test_param3.c -o test_param3 -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 6

struct thread_data{
  int thread_id;
  int sum;
  char *message;
};

/********************/
void *PrintHello(void *threadarg)
{
  long taskid, sum;
  struct thread_data *my_data;
  char *message;

  my_data = (struct thread_data *) threadarg;
  taskid = my_data->thread_id;
  sum = my_data->sum;
  message = my_data->message;
  
  printf("in PrintHello(), thread #%lx ; sum = %ld, message = %s\n",
          taskid, sum, message);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  char *Messages[NUM_THREADS] = {"First Message",
                                 "Second Message",
                                 "Third Message",
                                 "Fourth Message",
                                 "Fifth Message",
                                  "Sixth Message"};

  printf("Hello test_param3.c\n");

  for (long t = 0; t < NUM_THREADS; t++) {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].sum = t+27;
    thread_data_array[t].message = Messages[t];
    thread_data_array[5].sum = 1000;

    printf("In main:  creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, PrintHello,
                                             (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  pthread_exit(NULL);

} /* end main */