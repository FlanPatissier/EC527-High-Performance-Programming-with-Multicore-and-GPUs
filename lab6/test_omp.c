/********************************************************************

To compile:

  gcc -O1 -fopenmp test_omp.c -lrt -lm -o test_omp


To run:

Prefix your command with a directive to set the environment variable
OMP_NUM_THREADS, like this:

     OMP_NUM_THREADS=10 ./test_omp

It will detect the requested number of threads and use it.


*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* This define is only used if you do not set the environment variable
   OMP_NUM_THREADS as instructed above, and if OpenMP also does not
   automatically detect the hardware capabilities. */
#define THREADS 4

void detect_threads_setting()
{
  long int i, ognt;
  char *env_ONT;

  /* Find out how many threads OpenMP thinks it is wants to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++)
  {
    ognt = omp_get_num_threads();
  }

  printf("omp's default number of threads is %d\n", ognt);

  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0)
  {
    if (THREADS != ognt)
    {
      printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }

  omp_set_num_threads(ognt);

  /* Once again ask OpenMP how many threads it is going to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++)
  {
    ognt = omp_get_num_threads();
  }
  printf("Using %d threads for OpenMP\n", ognt);
}

/*********************************************************************/
int main(int argc, char *argv[])
{
  int ognt;

  printf("OpenMP race condition print test\n");

  detect_threads_setting();

  printf("Printing 'Hello world!' using 'omp parallel' and 'omp sections':\n\n    ");

#pragma omp parallel
#pragma omp sections
  {
    //    printf("\n");
    //#pragma omp section
    printf("H");
#pragma omp section
    printf("e");
#pragma omp section
    printf("l");
#pragma omp section
    printf("l");
#pragma omp section
    printf("o");
#pragma omp section
    printf(" ");
#pragma omp section
    printf("W");
#pragma omp section
    printf("o");
#pragma omp section
    printf("r");
#pragma omp section
    printf("l");
#pragma omp section
    printf("d");
#pragma omp section
    printf("!");
  }

  printf("\n\n");

  printf("Printing 'Hello world!' using 'omp parallel for':\n\n    ");

  /* ============ ADD YOUR CODE HERE ============ */
  int i;
  char string[12] = "Hello World!";
  //{"H","e","l","l", "o"," ","W", "o","r","l","d","!"};

#pragma omp parallel for

  for (i = 0; i < 12; i++)
  {
    printf("%c", string[i]);
  }

  printf("\n");

  return 0;
} /* end main */

/**********************************************/