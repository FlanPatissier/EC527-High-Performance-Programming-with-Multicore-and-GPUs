/*************************************************************************

  gcc -pthread test_crit.c -o test_crit -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 10000
pthread_mutex_t mutexA; 

struct thread_data{
  int thread_id;
  int *balance;
  double qr;
};

double qr_total = 0;

/********************/
void *PrintHello(void *threadarg)
{
  long int taskid;
  struct thread_data *my_data;
  int balance;
  double qr;

  my_data = (struct thread_data *) threadarg;
  taskid = my_data->thread_id;
  qr = my_data->qr;

  pthread_mutex_lock (&mutexA);


  /* get the old balance */
  balance = *(my_data->balance);


  /* Add up our quasi-random numbers, quasi-random delay, and add again */
  qr_total += qr;
  while (qr > 0.1) {
    qr = qr * 0.99;
  }
  qr_total += qr;
  /* (global qr_total gets printed at the end by main, to prevent compiler
     optimisation of the delay calculation we just did */



  /* Modify balance */
  if (taskid % 2 == 0) {
    balance += 1;
  } else {
    balance -= 1;  
  }

  /* write new balance to global */
  *(my_data->balance) = balance;
  pthread_mutex_unlock (&mutexA);


  /* printf(" It's me, thread #%ld! balance = %d\n", taskid, *balance); */

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS];
  struct thread_data thread_data_array[NUM_THREADS];
  int rc;
  int account = 1000;
  double quasi_random = 0;


  printf("Hello test_crit.c\n");

  pthread_mutex_init(&mutexA, NULL);

  for (long t = 0; t < NUM_THREADS; t++) {
    quasi_random = quasi_random*quasi_random - 1.923432;
    thread_data_array[t].thread_id = t+1;
    thread_data_array[t].balance = &account;
    thread_data_array[t].qr = quasi_random + 2.0;
    /* printf("In main:  creating thread %ld\n", t+1); */
    rc = pthread_create(&threads[t], NULL, PrintHello,
                                           (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
  pthread_mutex_destroy(&mutexA);


  for (long t = 0; t < NUM_THREADS; t++) {
    if (pthread_join(threads[t],NULL)){
      exit(19);
    }
  }

  printf(" MAIN --> final balance = %d\n", account); 
  printf("   qr_total = %f\n", qr_total); 


  pthread_exit(NULL);
} /* end main */