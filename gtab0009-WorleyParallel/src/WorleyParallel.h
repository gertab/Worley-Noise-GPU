#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include "jbutil.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Convert 3D position to the corresponding flattened 1D array position
#define position3D(x, y, z, WIDTH, HEIGHT) (HEIGHT*WIDTH*z + WIDTH*y + x)
//#define position2D(x, y, WIDTH) (WIDTH*y + x)

// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

void printHelp(char *input);
void randomPointGeneration(int *random_points_x, int* random_points_y, jbutil::randgen rand, int tile_x, int tile_y, int tile_size, int points_per_tile);
void WorleyNoise(const std::string outfile, const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory);
void PerformanceCheck(const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory);


__global__ void normDistanceFromNearestPoint(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result);
__global__ void normDistanceFromNearestPointSharedMemory(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result);


__host__ __device__
static void print3DMatrix(int *data, int width, int height, int depth) {

	printf("Printing data\n");
	for(int z = 0; z < depth; z++) {
		printf("\ndepth: %d\n", z);

		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				printf("%10d ", data[position3D(x, y, z, width, height)]);
			}

			printf("\n");
		}
	}
}
