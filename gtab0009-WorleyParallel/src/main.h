#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include "jbutil.h"

// Macros
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchkRand(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


// Convert 3D position to the corresponding flattened 1D array position
#define position3D(x, y, z, WIDTH, HEIGHT, DEPTH) (DEPTH * WIDTH * y + DEPTH * x + z)
#define position2D(x, y, WIDTH, HEIGHT) (WIDTH * y + x)

// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

enum MEMORY {
	shared,
	constant,
	naive
};
void printHelp(char *input);
void WorleyNoise(const std::string outfile, const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory, bool fast_math = false);
void PerformanceCheck(const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory, const bool fast_math);

