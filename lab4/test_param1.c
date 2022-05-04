/*************************************************************************

  gcc -pthread test_param1.c -o test_param -std=gnu99

 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

/************************************************************************/
void *PrintHello(void *threadid)
{
    long tid;

    tid = (long) threadid;

    // signed char tid;
    
    // tid = (signed char) threadid;
    
    printf("PrintHellos() in thread # %ld ! \n", tid);
    
    pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    printf("Hello test_param1.c\n");
    
    for (t = 0; t < NUM_THREADS; t++) {
        printf("In main:  creating thread %d\n", t);
        // by using (void*) t, we pass a value
        signed char u = (signed char) (-1 * t);
        rc = pthread_create(&threads[t], NULL, PrintHello, (void*) u);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    printf("It's me MAIN -- Good Bye World!\n");

    pthread_exit(NULL);

} /* end main */