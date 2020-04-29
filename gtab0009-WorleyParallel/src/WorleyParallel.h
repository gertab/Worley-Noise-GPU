#include "jbutil.h"

__global__ void normDistanceFromNearestPoint(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result);
__global__ void normDistanceFromNearestPointSharedMemory(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result, bool fast_math = false);
void randomPointGeneration(int *random_points_x, int *random_points_y, jbutil::randgen rand, int tile_x, int tile_y, int tile_size, int points_per_tile);
