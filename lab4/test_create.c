/**************************************************************

   gcc -pthread test_create.c -o test_create -std=gnu99

 */

/* Simple thread create and exit test */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 5

/***************************************************************/
void *work(void *i)
{
  printf("Hello World! from child thread %lx\n", (long)pthread_self());
  // sleep(3);

  pthread_exit(NULL);
}

/****************************************************************/
int main(int argc, char *argv[])
{
  int arg,j,k,m;                  /* Local variables. */
  // pthread_t id[NUM_THREADS];
  pthread_t *id = (pthread_t*) malloc(sizeof(pthread_t) * NUM_THREADS);


  printf("Hello test_create.c\n");

  for (long t = 0; t < NUM_THREADS; ++t) {
    if (pthread_create(&id[t], NULL, work, NULL)) {
      printf("ERROR creating the thread\n");
      exit(19);
    }
  }

  printf("main() after creating the thread.  My id is %lx\n",(long) pthread_self());
  sleep(3);
  return(0);
} /* end main */