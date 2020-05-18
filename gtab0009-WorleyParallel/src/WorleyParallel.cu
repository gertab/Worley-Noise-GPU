//============================================================================
// Name        : main.cu
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation
//============================================================================

#include "jbutil.h"
#include "WorleyParallel.h"
#include "main.h"
#include <curand_kernel.h>

// Creates Worley Noise depending on the options
void WorleyNoise(const std::string outfile, const int width, const int height, const int tile_size,
		const int points_per_tile, const float intensity, int seed, const bool reverse, const bool shared_memory, const bool fast_math) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);

	if(seed == 0)
		seed = time(NULL); // Random seed

	std::cout << "Creating Worley Noise using the GPU, with size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << ", fast math: " << (fast_math ? "yes" : "no") << "\n";

	// start timer
	double t = jbutil::gettime();

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);
	jbutil::randgen rand(seed);

	// Random points
	size_t random_points_size = tile_x * tile_y * points_per_tile * sizeof(int);

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	// Allocating memory on device
	size_t res_size = width * height * sizeof(int);
	int *d_result, *d_random_points_x, *d_random_points_y;
	gpuErrchk( cudaMalloc((void**) &d_result, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_x, random_points_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_y, random_points_size) );

	// Generate random number directly on device
	generateRandomPointOnDevice(d_random_points_x, d_random_points_y, seed, tile_x, tile_y, tile_size, points_per_tile);

	dim3 grid(DIV_CEIL(width, 32), DIV_CEIL(height, 32));
	dim3 blocks(32, 32);

	if(shared_memory) {
		std::cout << "Using shared memory";
		if(fast_math) {
			std::cout << " and fast math";
		}
		std::cout << "\n";

	    int sharedMemory = 2 * 9 * points_per_tile * sizeof(int); // 2: {x, y}, 9: neigbouring tiles
    	normDistanceFromNearestPointSharedMemory<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result, fast_math);
	} else  {
		std::cout << "Without using shared memory\n";
		normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);
	}

//    gpuErrchk( cudaPeekAtLastError() );
//    gpuErrchk( cudaDeviceSynchronize() );

    // Copy result back to host from device
	int *result = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result, d_result, res_size, cudaMemcpyDeviceToHost) );

	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			image_out(0, y, x) = result[position2D(x, y, width, height)];
			if(reverse) {
				// Reverse image: white -> black, black -> white
				image_out(0, y, x) = 255 - image_out(0, y, x);
			}
		}
	}

	// stop timer
	t = jbutil::gettime() - t;

	// save image
	std::ofstream file_out(outfile.c_str());
	image_out.save(file_out);
	// show time taken
	std::cerr << "Total time taken: " << t << "s" << std::endl;
}

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

						float x_point = random_points_x[position3D(i, j, k, tile_x, tile_y, points_per_tile)];
						float y_point = random_points_y[position3D(i, j, k, tile_x, tile_y, points_per_tile)];
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

	result[position2D(x, y, width, height)] = shortest_norm_dist;
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

		int position_shared_memory_1D = position3D(shared_memory_x, shared_memory_y, shared_memory_z, 3, 3, points_per_tile);
		if(shared_tile_x_pos >= 0 && shared_tile_x_pos < tile_x
				&& shared_tile_y_pos >= 0 && shared_tile_y_pos < tile_y) {
			// Tile is within range
		    tiles_x[position_shared_memory_1D] = random_points_x[position3D(shared_tile_x_pos, shared_tile_y_pos, shared_memory_z, tile_x, tile_y, points_per_tile)];
			tiles_y[position_shared_memory_1D] = random_points_y[position3D(shared_tile_x_pos, shared_tile_y_pos, shared_memory_z, tile_x, tile_y, points_per_tile)];
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

				float x_point = tiles_x[position3D(i, j, k, 3, 3, points_per_tile)];
				float y_point = tiles_y[position3D(i, j, k, 3, 3, points_per_tile)];

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

	result[position2D(x, y, width, height)] = shortest_norm_dist;
}

// Generates random points on the host
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
				random_points_x[position3D(x, y, z, tile_x, tile_y, points_per_tile)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				random_points_y[position3D(x, y, z, tile_x, tile_y, points_per_tile)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);
			}
		}
	}
}

// Generates random points on the device
void generateRandomPointOnDevice(int *d_random_points_x, int *d_random_points_y, int seed, int tile_x, int tile_y, int tile_size, int points_per_tile) {
	dim3 grid(DIV_CEIL(tile_x, 16), DIV_CEIL(tile_y, 16));
	dim3 blocks(16, 16);

	generateRandomPointsKernel<<<grid, blocks>>>(d_random_points_x, d_random_points_y, seed, tile_size, tile_x, tile_y, points_per_tile);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void generateRandomPointsKernel2(int *d_random_points_x, int *d_random_points_y, int seed, int tile_size, int tile_x, int tile_y, int points_per_tile){
	assert(d_random_points_x != nullptr && d_random_points_y != nullptr);
	assert(tile_x > 0 && tile_y > 0);
	assert(tile_size > 0);
	assert(points_per_tile > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= tile_x || y >= tile_y) {
    	// Out of bounds
    	return;
    }

    int id = position3D(x, y, 0, tile_x, tile_y, points_per_tile);

    float r;
    curandState state;

	curand_init(seed, id, 0, &state);

	for(int z = 0; z < points_per_tile; z++) {

		r = curand_uniform(&state); // Random number from uniform distribution

		int a = x * tile_size, b = (x + 1) * tile_size;
		int x_res = r * (b - a) + a;

		r = curand_uniform(&state);

		a = y * tile_size;
		b = (y + 1) * tile_size;
		int y_res = r * (b - a) + a;

		d_random_points_x[position3D(x, y, z, tile_x, tile_y, points_per_tile)] = x_res;
		d_random_points_y[position3D(x, y, z, tile_x, tile_y, points_per_tile)] = y_res;
	}
}

// Kernel to generate random points using cuRAND
__global__ void generateRandomPointsKernel(int *d_random_points_x, int *d_random_points_y, int seed, int tile_size, int tile_x, int tile_y, int points_per_tile){
	assert(d_random_points_x != nullptr && d_random_points_y != nullptr);
	assert(tile_x > 0 && tile_y > 0);
	assert(tile_size > 0);
	assert(points_per_tile > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= tile_x || y >= tile_y) {
    	// Out of bounds
    	return;
    }

    int id = position3D(x, y, 0, tile_x, tile_y, points_per_tile);

    float r;
    curandState state;

	curand_init(seed, id, 0, &state);

	for(int z = 0; z < points_per_tile; z++) {

		r = curand_uniform(&state); // Random number from uniform distribution

		int a = x * tile_size, b = (x + 1) * tile_size;
		int x_res = r * (b - a) + a;

		r = curand_uniform(&state);

		a = y * tile_size;
		b = (y + 1) * tile_size;
		int y_res = r * (b - a) + a;

		d_random_points_x[position3D(x, y, z, tile_x, tile_y, points_per_tile)] = x_res;
		d_random_points_y[position3D(x, y, z, tile_x, tile_y, points_per_tile)] = y_res;
	}
}

// Performance checking for Worley Noise
// Timings exclude memory transfer to/from device
void PerformanceCheck(const int width, const int height, const int tile_size, const int points_per_tile, const float intensity,
		int seed, const bool reverse, const bool shared_memory, const bool fast_math) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);

	if(seed == 0)
		seed = time(NULL); // Random seed

	std::cout << "Performance testing Worley Noise using the GPU, with size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << ", fast math: " << (fast_math ? "yes" : "no") << "\n";

	if(shared_memory) {
		std::cout << "Using shared memory\n";
	} else {
		std::cout << "Without using shared memory\n";
	}

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);

	jbutil::randgen rand(seed);

	// Random points
	size_t random_points_size = tile_x * tile_y * points_per_tile * sizeof(int);
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	// Allocating memory on device
	size_t res_size = width * height * sizeof(int);
	int *d_result, *d_random_points_x, *d_random_points_y;
	gpuErrchk( cudaMalloc((void**) &d_result, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_x, random_points_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_y, random_points_size) );

	// start timer
	double t = jbutil::gettime();

	int count = 0;

	// Loop for at least 60 seconds
	while((jbutil::gettime() - t) < 60) {
		count++;

		if(true) {
			// Generate random number directly on device
			generateRandomPointOnDevice(d_random_points_x, d_random_points_y, seed, tile_x, tile_y, tile_size, points_per_tile);

		} else {
			// Generate random number on the host and transfer results to device
			random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
			random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
			// Generate random points
			randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

			// Copying data to device
			gpuErrchk( cudaMemcpy(d_random_points_x, random_points_x, random_points_size, cudaMemcpyHostToDevice) );
			gpuErrchk( cudaMemcpy(d_random_points_y, random_points_y, random_points_size, cudaMemcpyHostToDevice) );

			free(random_points_x);
			free(random_points_y);
		}

		int block_width = 32;
		dim3 grid(DIV_CEIL(width, block_width), DIV_CEIL(height, block_width));
		dim3 blocks(block_width, block_width);

		if(shared_memory) {
		    int sharedMemory = 2 * 9 * points_per_tile * sizeof(int);
		    normDistanceFromNearestPointSharedMemory<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);
		} else  {
			normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);
		}

		gpuErrchk( cudaDeviceSynchronize() );
	}

	// stop timer
	t = jbutil::gettime() - t;

	// show time taken
	std::cout << "\n\n";
	std::cerr << "Ran " << count << " iterations in " << t << "s. Average time taken: " << t / count << " s" << std::endl;
}
