//============================================================================
// Name        : main.cu
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation
//============================================================================


#include "jbutil.h"
#include "WorleyParallel.h"
#include "main.h"


// Works the normalized distances from closest pixel (x, y) to the nearest point from (random_point_x, random_point_y)
__global__ void normDistanceFromNearestPoint(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result) {
	assert(tile_size > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) {
    	return;
    }

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);

	// 0   = black
	// 255 = white
	int shortest_norm_dist = 255;

	// Check only 3 by 3 tiles closest to current pixel
	// This avoid having to brute force all points
	for(int i = tile_x_pos - 1; i <= tile_x_pos + 1; i++) {
		if(i >= 0 && tile_x_pos < tile_x) {

			for(int j = tile_y_pos - 1; j <= tile_y_pos + 1; j++) {
				if(j >= 0 && tile_y_pos < tile_y) {
					// Ensure tile is within range

					for(int k = 0 ; k < points_per_tile ; k++){
						// Checking all points in current tile

						float x_point = random_points_x[position3D(i, j, k, tile_x, tile_y)];
						float y_point = random_points_y[position3D(i, j, k, tile_x, tile_y)];
						float x_dist = (x - x_point) / intensity;
						float y_dist = (y - y_point) / intensity;

						int distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euclidean distance

						if(distance < shortest_norm_dist) {
							shortest_norm_dist = distance;
						}
					}
				}
			}
		}
	}

	result[position3D(x, y, 0, width, height)] = shortest_norm_dist;
}

// Works the normalized distances from closest pixel (x, y) to the nearest point from (random_point_x, random_point_y)
// Using shared memory
__global__ void normDistanceFromNearestPointSharedMemory(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result, bool fast_math) {
	assert(tile_size > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);

	// Load to shared memory
	extern __shared__ int s[];
	int *tiles_x = s;
	int *tiles_y = (int*) &tiles_x[9 * points_per_tile];

	// Each thread in one block is assigned different index;
	int indexInBlock = threadIdx.x + blockDim.x * threadIdx.y;

	assert((blockDim.x * blockDim.y) >= 9 * (points_per_tile));

	if(indexInBlock < (9 * points_per_tile)) {
		int tileToGet = indexInBlock / points_per_tile;
		int shared_memory_z = indexInBlock % points_per_tile;

		// shared_memory_x can have value in range [0, 2]. Same for shared_memory_y
		int shared_memory_x = tileToGet % 3;
		int shared_memory_y = tileToGet / 3;

		// Convert from ranges [0, 2] to {shared_memory_x - 1, shared_memory_x, shared_memory_x + 1}. Same for shared_memory_y
		int shared_tile_x_pos = shared_memory_x + tile_x_pos - 1;
		int shared_tile_y_pos = shared_memory_y + tile_y_pos - 1;

		int position_shared_memory_1D = position3D(shared_memory_x, shared_memory_y, shared_memory_z, 3, 3);
		if(shared_tile_x_pos >= 0 && shared_tile_x_pos < tile_x
				&& shared_tile_y_pos >= 0 && shared_tile_y_pos < tile_y) {
			// Tile is within range
		    tiles_x[position_shared_memory_1D] = random_points_x[position3D(shared_tile_x_pos, shared_tile_y_pos, shared_memory_z, tile_x, tile_y)];
			tiles_y[position_shared_memory_1D] = random_points_y[position3D(shared_tile_x_pos, shared_tile_y_pos, shared_memory_z, tile_x, tile_y)];
		} else {
			// Tiles out of range are zeroed
			tiles_x[position_shared_memory_1D] = -1;
			tiles_y[position_shared_memory_1D] = -1;
		}
	}

    if(x >= width || y >= height) {
    	// x and y  bigger than the limits are still used to load data in shared memory, since that if the tiles (in the right/bottom border)
    	// are bigger than the block size, they would still be needed
    	return;
    }

    // Ensure shared memory is filled
    __syncthreads();

	// 0   = black
	// 255 = white
	int shortest_norm_dist = 255;

	// Check only 3 by 3 tiles closest to current pixel
	// This avoid having to brute force all points
	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			for(int k = 0 ; k < points_per_tile; k++){
				// Checking all points in current tile

				float x_point = tiles_x[position3D(i, j, k, 3, 3)];
				float y_point = tiles_y[position3D(i, j, k, 3, 3)];

				if(!(x_point == -1 && y_point == -1)) {
					float x_dist, y_dist, distance;

					if(fast_math) {
						// Use faster but less accurate math functions
						x_dist = __fdividef(__fsub_rd(x, x_point), intensity);
						y_dist = __fdividef(__fsub_rd(y, y_point), intensity);

						distance = __fsqrt_rd(__fadd_rd(__fmul_ru(x_dist, x_dist), __fmul_ru(y_dist, y_dist))); // Euclidean distance
					} else {
						// Use accurate math functions

						x_dist = (x - x_point) / intensity;
						y_dist = (y - y_point) / intensity;

						distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euclidean distance
					}

					if(distance < shortest_norm_dist) {
						shortest_norm_dist = (int) distance;
					}
				}
			}
		}
	}

	result[position3D(x, y, 0, width, height)] = shortest_norm_dist;
}


// Fills random_points_x and random_points_y with random numbers
// random_points_x and random_points_y should have enough space to be filled with (tile_x * tile_y * points_per_tile) random numbers
void randomPointGeneration(int *random_points_x, int *random_points_y, jbutil::randgen rand, int tile_x, int tile_y, int tile_size, int points_per_tile) {
	assert(random_points_x != nullptr && random_points_y != nullptr);
	assert(tile_x > 0 && tile_y > 0);
	assert(tile_size > 0);
	assert(points_per_tile > 0);

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int z = 0; z < points_per_tile; z++) {
				random_points_x[position3D(x, y, z, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				random_points_y[position3D(x, y, z, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);
			}
		}
	}
}
