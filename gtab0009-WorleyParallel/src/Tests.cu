//============================================================================
// Name        : Tests.cpp
// Author      : Gerard Tabone
// Version     :
// Description : Tests for worley noise
//============================================================================

#include "Tests.h"
#include "WorleyParallel.h"

void runTests() {
	std::cout << "Running test cases\n";
	test1();
	test2();

//	testDIVCEIL();
//	testPosition3D();
	std::cout << "End of tests. No problems found.\n";
}

void test1() {
	// Testing small dataset, with image 3 x 3

	// Default
	int width = 3;
	int height = 3;
	int tile_size = 32;
	int points_per_tile = 1;
	float intensity = 1;
	int seed = 123;
	bool shared_memory = true;

	int tile_x = DIV_CEIL(width, tile_size);
	assert(tile_x == 1);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_y == 1);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	// Ensuring constant random point: (0, 0)
	random_points_x[0] = 0;
	random_points_y[0] = 0;

	// Correct output
	int correct_result[3][3] = {{0, 1, 2},
								{1, 1, 2},
								{2, 2, 2}};

	size_t random_points_size = tile_x * tile_y * points_per_tile * sizeof(int);

	// Allocating memory on device
	size_t res_size = width * height * sizeof(int);
	int *d_result_no_shared_mem, *d_result_shared_mem, *d_random_points_x, *d_random_points_y;
	gpuErrchk( cudaMalloc((void**) &d_result_no_shared_mem, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_result_shared_mem, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_x, random_points_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_y, random_points_size) );

	// Copying data to device
	gpuErrchk( cudaMemcpy(d_random_points_x, random_points_x, random_points_size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_random_points_y, random_points_y, random_points_size, cudaMemcpyHostToDevice) );

	dim3 grid(DIV_CEIL(width, 32), DIV_CEIL(height, 32));
	dim3 blocks(32, 32);

		// Shared memory
	    int sharedMemory = 2 * 9 * points_per_tile * sizeof(int);
	    normDistanceFromNearestPointSharedMemory<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result_shared_mem);

		// No shared memory
		normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result_no_shared_mem);


    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy result back to host from device
	int *result_no_shared_mem = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result_no_shared_mem, d_result_no_shared_mem, res_size, cudaMemcpyDeviceToHost) );
	int *result_shared_mem = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result_shared_mem, d_result_shared_mem, res_size, cudaMemcpyDeviceToHost) );

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int point_no_shared = result_no_shared_mem[position3D(x, y, 0, width, height)];
			int point_shared = result_shared_mem[position3D(x, y, 0, width, height)];

			assert(correct_result[y][x] == point_no_shared);
			assert(correct_result[y][x] == point_shared);
		}
	}


	free(random_points_x);
	free(random_points_y);
	free(result_no_shared_mem);
	free(result_shared_mem);

}

void test2() {
	// Testing small dataset, with image 4 x 4 having two point per tile

	// Default
	int width = 4;
	int height = 4;
	int tile_size = 32;
	int points_per_tile = 2;
	float intensity = 1;
	int seed = 123;

	int tile_x = DIV_CEIL(width, tile_size);
	assert(tile_x == 1);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_y == 1);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	// Ensuring constant random point: (0, 0) and (3, 3)
	random_points_x[0] = 0;
	random_points_y[0] = 0;
	random_points_x[1] = 3;
	random_points_y[1] = 3;

	// Correct output
	int result[4][4] = {{0, 1, 2, 3},
						{1, 1, 2, 2},
						{2, 2, 1, 1},
						{3, 2, 1, 0}};

	int i = 0;
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
		   int p = normDistanceFromNearestPointSerialImplementation(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);

		   assert(result[y][x] == p);
		   i++;
		}
	}

	free(random_points_x);
	free(random_points_y);
}

void test3() {
	// Testing larger dataset, with image 10 x 10

	// Default
	int width = 10;
	int height = 10;
	int tile_size = 32;
	int points_per_tile = 1;
	float intensity = 1;
	int seed = 768;

	int tile_x = DIV_CEIL(width, tile_size);
	assert(tile_x == 1);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_y == 1);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

	// Ensuring constant random points;
	assert(random_points_x[0] == 22);
	assert(random_points_y[0] == 1);

	// Expected output
	int result[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 3, 3, 3, 4, 5, 5, 6, 7, 8, 9, 4, 4, 4, 5, 5, 6, 7, 8, 8, 9, 5, 5, 5, 5, 6, 7, 7, 8, 9, 10, 6, 6, 6, 6, 7, 7, 8, 9, 10, 10, 7, 7, 7, 7, 8, 8, 9, 9, 10, 11, 8, 8, 8, 8, 8, 9, 10, 10, 11, 12, 9, 9, 9, 9, 9, 10, 10, 11, 12, 12};
	int i = 0;
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
		   int p = normDistanceFromNearestPointSerialImplementation(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);

		   assert(result[i] == p);
		   i++;
		}
	}

	free(random_points_x);
	free(random_points_y);
}

//void testDIVCEIL() {
//	// Test division and ceiling function
//
//	assert(DIV_CEIL(0, 5) == 0);
//	assert(DIV_CEIL(1, 5) == 1);
//	assert(DIV_CEIL(2, 5) == 1);
//	assert(DIV_CEIL(3, 5) == 1);
//	assert(DIV_CEIL(4, 5) == 1);
//	assert(DIV_CEIL(5, 5) == 1);
//	assert(DIV_CEIL(6, 5) == 2);
//	assert(DIV_CEIL(7, 5) == 2);
//	assert(DIV_CEIL(8, 5) == 2);
//}
//
//void testPosition3D() {
//	int width = 2, height = 2;
//
//	assert(position3D(0, 0, 0, width, height) == 0);
//	assert(position3D(1, 0, 0, width, height) == 1);
//	assert(position3D(0, 1, 0, width, height) == 2);
//	assert(position3D(1, 1, 0, width, height) == 3);
//	assert(position3D(0, 0, 1, width, height) == 4);
//	assert(position3D(1, 0, 1, width, height) == 5);
//	assert(position3D(0, 1, 1, width, height) == 6);
//	assert(position3D(1, 1, 1, width, height) == 7);
//}


// Serial implementation
// Works the normalized distances from closest pixel (x, y) to the nearest point from (random_point_x, random_point_y)
int normDistanceFromNearestPointSerialImplementation(int x, int y, int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity) {

	assert(tile_size > 0);
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

	return shortest_norm_dist;
}
