/*************************************************************************/
// gcc -pthread test_sync2.c -o test_sync2 -std=gnu99

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 8

struct thread_data{
  int thread_id;
  int sum;
  char *message;
};

pthread_mutex_t mutexA[NUM_THREADS+1];

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

  /* printf("PrintHello thread #%ld, sum = %ld, message = %s\n",
                                       taskid, sum, message); */

  switch (taskid) {
  case 2:
    printf("thread #%ld waiting for 1 ...\n", taskid);
    pthread_mutex_lock(&mutexA[1]);
    pthread_mutex_unlock(&mutexA[1]);
    break;
  case 3:
    printf("thread #%ld waiting for 1 ...\n", taskid);
    pthread_mutex_lock(&mutexA[1]);   
    pthread_mutex_unlock(&mutexA[1]);
    break;
  case 4:
    printf("thread #%ld waiting for 2 ...\n", taskid);
    pthread_mutex_lock(&mutexA[2]);   
    pthread_mutex_unlock(&mutexA[2]);
    break;
  case 5:
    printf("thread #%ld waiting for 4 ...\n", taskid);
    pthread_mutex_lock(&mutexA[4]);
    pthread_mutex_unlock(&mutexA[4]);
    break;
  case 6:
    printf("thread #%ld waiting for 3 and 4 ...\n", taskid);
    pthread_mutex_lock(&mutexA[3]);
    pthread_mutex_unlock(&mutexA[3]);
    pthread_mutex_lock(&mutexA[4]);
    pthread_mutex_unlock(&mutexA[4]);
    break;
  case 7:
    printf("thread #%ld waiting for 5 and 6 ...\n", taskid);
    pthread_mutex_lock(&mutexA[5]);
    pthread_mutex_unlock(&mutexA[5]);
    pthread_mutex_lock(&mutexA[6]);
    pthread_mutex_unlock(&mutexA[6]);
    break;
  case 8:
	  printf("thread #%ld waiting for 6 and 7 ...\n", taskid);
	  pthread_mutex_lock(&mutexA[6]);
	  pthread_mutex_unlock(&mutexA[6]);
	  pthread_mutex_lock(&mutexA[7]);
	  pthread_mutex_unlock(&mutexA[7]);
	  break;

  default:
    printf("It's me, thread #%ld! I'm waiting ...\n", taskid);
    break;
  }
  printf("It's me, thread #%ld! I'm unlocking %ld...\n", taskid, taskid);

  if (pthread_mutex_unlock(&mutexA[taskid])) {
    printf("ERROR on unlock\n");
  }

  /* printf("thread #%ld done\n", taskid); */

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS+1];
  struct thread_data thread_data_array[NUM_THREADS+1];
  int rc;
  char *Messages[NUM_THREADS+1] = {"Zeroth Message",
                                 "First Message",
                                 "Second Message",
                                 "Third Message",
                                 "Fourth Message",
                                 "Fifth Message",
                                 "Sixth Message",
                                 "Seventh Message",
                                 "Eighth Message"};
  char dummy[1];

  for (long t = 0; t <= NUM_THREADS; t++) {
    if (pthread_mutex_init(&mutexA[t], NULL)) {
      printf("ERROR initializing the locks\n");
      exit(-1);
    }
  }

  for (long t = 0; t <= NUM_THREADS; t++) {
    if (pthread_mutex_lock(&mutexA[t])) {
      printf("ERROR on lock\n");
      exit(-1);
    }
  }

  printf("Hello test_sync2.c\n");

  for (long t = 2; t <= NUM_THREADS; t++) {
    thread_data_array[t].thread_id = t;
    thread_data_array[t].sum = t+28;
    thread_data_array[t].message = Messages[t];
    /*  printf("In main:  creating thread %ld\n", t); */
    rc = pthread_create(&threads[t], NULL, PrintHello,
                                    (void*) &thread_data_array[t]);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  printf("Main: calling sleep(1)...\n");
  sleep(1);
  printf("Main: created threads 2-8, type a letter (not space) and <enter>\n");

  scanf("%s", dummy);

  printf("Main: waiting for thread 8 to finish, UNLOCK LOCK 1\n");
  
  if (pthread_mutex_unlock(&mutexA[1])) {
    printf("ERROR on unlock\n");
  }

  // unlock mutexA[1]
  printf("Main: Done unlocking 1 \n");
  
  // lock mutexA[7]
  pthread_mutex_lock(&mutexA[8]);

  for (long t = 2; t <= NUM_THREADS; t++) {
    if (pthread_join(threads[t], NULL)) {
      printf(" ERROR on join\n");
      exit(19);
    }
  }
  printf("Main: After joining\n");

  return(0);
} /* end main */