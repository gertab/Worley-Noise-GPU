#include "jbutil.h"

void WorleyNoise(const std::string outfile, const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory, bool fast_math = false);
void PerformanceCheck(const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory, const bool fast_math);

__global__ void normDistanceFromNearestPoint(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result);
__global__ void normDistanceFromNearestPointSharedMemory(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result, bool fast_math = false);

void randomPointGeneration(int *random_points_x, int *random_points_y, jbutil::randgen rand, int tile_x, int tile_y, int tile_size, int points_per_tile);

void generateRandomPointOnDevice(int *d_random_points_x, int *d_random_points_y, int seed, int tile_x, int tile_y, int tile_size, int points_per_tile);
__global__ void generateRandomPointsKernel(int *d_random_points_x, int *d_random_points_y, int seed, int tile_size, int tile_x, int tile_y, int points_per_tile);
