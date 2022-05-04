/*

  gcc -O1 -mavx -std=gnu99 avx_align.c -o avx_align

 */

#include <stdio.h>
#include <immintrin.h>

void unalign_heap_naive(float * a);
void align_heap_1(float * a);
void unalign_storeu_ps(float * a);
void unalign_local_alloc(void);

int main(int argc, char *argv[])
{
  float * a;
  long int ia;

  printf("AVX load/store alignment tests\n");

  unalign_local_alloc();

  // the size of a is 32+400=432
  a = (float *) malloc(32 + (100 * sizeof(float)));
  ia = (long int) a;
  // ia位于第一个能够整除32的地方
  for (int j=0; j<32 && ((ia % 32) != 0); j++) { ia++; }
  a = (float *) ia;


  // unalign_heap_naive(a);
  align_heap_1(a);
  unalign_storeu_ps(a);

  return 0;
} /* end main */

/* Unaligned writes on an array passed in by pointer, using naive
   "*my_pointer = my_m256_variable" approach. */
void unalign_heap_naive(float *a)
{
  __m256* p0 = (__m256*) &a[0];
  __m256* p1 = (__m256*) &a[1];
  __m256* p2 = (__m256*) &a[2];
  __m256 m1, m2;

  printf("unalign_heap_naive:\n");
  printf("  p1 == 0x%lx   p2 == 0x%lx\n", (long int) p1, (long int) p2);
  for(int i=0; i<100; i++) {
    a[i] = i+2;
  }
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m1 = *p0;
  *p1 = m1;
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m2 = _mm256_sqrt_ps(m1);
  *p2 = m2;
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  printf("\n");
}

/* Fix "*ptr = var" by just making sure all the pointers are at ofsets of 32
   bytes (8 floats) from each other. */
void align_heap_1(float *a)
{
  __m256* p0 = (__m256*) &a[0];
  __m256* p1 = (__m256*) &a[8];
  __m256* p2 = (__m256*) &a[16];
  __m256 m1, m2;

  printf("align_heap_1:\n");
  printf("  p1 == 0x%lx   p2 == 0x%lx\n", (long int) p1, (long int) p2);
  for(int i=0; i<100; i++) {
    a[i] = i+2;
  }
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m1 = *p0;
  *p1 = m1;
  for(int i=8; i<18; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m2 = _mm256_sqrt_ps(m1);
  *p2 = m2;
  for(int i=16; i<26; i++) { printf(" %6.4g", a[i]); } printf("\n");
  printf("\n");
}

/* Fix "*ptr = var" using _mm256_storeu_ps */
void unalign_storeu_ps(float *a)
{
  __m256* p0 = (__m256*) &a[0];
  float * p1 = &a[1];
  float * p2 = &a[2];
  __m256 m1, m2;

  printf("unalign_storeu_ps:\n");
  printf("  p1 == 0x%lx   p2 == 0x%lx\n", (long int) p1, (long int) p2);
  for(int i=0; i<100; i++) {
    a[i] = i+2;
  }
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m1 = *p0;
  // uses instruction with less stringent alignment requirement
  _mm256_storeu_ps(p1, m1);
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m2 = _mm256_sqrt_ps(m1);
  _mm256_storeu_ps(p2, m2);
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  printf("\n");
}

/* Try unaligned write on an array allocated locally on the stack. As
   of early 2021, this worked. */
void unalign_local_alloc(void)
{
  float a[100];
  __m256* p0 = (__m256*) &a[0];
  __m256* p1 = (__m256*) &a[1];
  __m256* p2 = (__m256*) &a[2];
  __m256 m1, m2;

  printf("unalign_local_alloc:\n");
  printf("  p1 == 0x%lx   p2 == 0x%lx\n", (long int) p1, (long int) p2);
  for(int i=0; i<100; i++) {
    a[i] = i+2;
  }
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m1 = *p0;
  *p1 = m1;
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  m2 = _mm256_sqrt_ps(m1);
  *p2 = m2;
  for(int i=0; i<10; i++) { printf(" %6.4g", a[i]); } printf("\n");
  printf("\n");
}