/*************************************************************************

 Simple thread create and exit test

   gcc -pthread test_join.c -o test_join -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 5

/*************************************************************************/
void *work(void *i)
{
  sleep(3);

  printf(" Hello World!  It's me, thread #%lx --\n", (long)pthread_self());

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  pthread_t id[NUM_THREADS];

  printf("Hello test_join.c\n");

  for (long t = 0; t < NUM_THREADS; t++) {
    if (pthread_create(&id[t], NULL, work, NULL)) {
      exit(19);
    }
  }

  printf("main() -- After creating the thread.  My id is: %lx\n",
                                                      (long)pthread_self());

  for (long t = 0; t < NUM_THREADS; t++) {
    if (pthread_join(id[t], NULL)) {
      exit(19);
    }
  }

  printf("After joining, Good Bye World!\n");

  return(0);
} /* end main */
