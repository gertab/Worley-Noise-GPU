//============================================================================
// Name        : WorleyParallel.cu
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include "jbutil.h"
#include "worleynoise.h"

__global__
void normDistanceFromNearestPoint(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int distance_order, int *result) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) {
    	return;
    }

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	// 0   = black
	// 255 = white
	int shortest_norm_dist = 255;
	int second_shortest_norm_dist = 255;

	// Check 3 by 3 tiles closest to current position
	// This avoid having to brute force all points
	for(int i = tile_x_pos - 1; i <= tile_x_pos + 1; i++) {
		if(i >= 0 && tile_x_pos < tile_x) {

			for(int j = tile_y_pos - 1; j <= tile_y_pos + 1; j++) {
				if(j >= 0 && tile_y_pos < tile_y) {
					// Ensure tile is within range

					for(int k = 0 ; k < points_per_tile ; k++){
						// Checking all points in current tile

						float x_point = random_points_x[position(i, j, k, tile_x, tile_y)];
						float y_point = random_points_y[position(i, j, k, tile_x, tile_y)];
						float x_dist = (x - x_point) / intensity;
						float y_dist = (y - y_point) / intensity;

//						int distance = abs(x_dist) + abs(y_dist); // Manhattan distance
						int distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euclidean distance

						if(distance < shortest_norm_dist) {
							second_shortest_norm_dist = shortest_norm_dist;
							shortest_norm_dist = distance;
						} else if(distance < second_shortest_norm_dist) {
							second_shortest_norm_dist = distance;
						}
					}
				}
			}
		}
	}


	if(distance_order == 1) {
		result[position2D(x, y, height)] = shortest_norm_dist;
	} else {
		result[position2D(x, y, height)] = second_shortest_norm_dist;
	}
}


__global__
void normDistanceFromNearestPointSharedMem(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int distance_order, int *result) {

	extern __shared__ int tiles_x[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	if(threadIdx.x < 9 * points_per_tile) {//)& threadIdx.y < 3 * points_per_tile) {
		int i = tile_x_pos - 1 + threadIdx.x;

		if(i >= 0 && tile_x_pos < tile_x) {

			int j = tile_y_pos - 1 + threadIdx.y;
			if(j >= 0 && tile_y_pos < tile_y) {

//				for(int k = 0 ; k < points_per_tile ; k++){

				int k = threadIdx.x / points_per_tile;
					tiles_x[position(threadIdx.x % 3, threadIdx.y % 3, k, 3, 3)] = random_points_x[position(i, j, k, tile_x, tile_y)];

//				}
			}
		}

	}

    if(x >= width || y >= height) {
    	return;
    }

	__syncthreads();

	// 0   = black
	// 255 = white
	int shortest_norm_dist = 255;
	int second_shortest_norm_dist = 255;

	// Check 3 by 3 tiles closest to current position
	// This avoid having to brute force all points
	for(int i = tile_x_pos - 1, u = 0; i <= tile_x_pos + 1; i++, u++) {
		if(i >= 0 && tile_x_pos < tile_x) {

			for(int j = tile_y_pos - 1, v = 0; j <= tile_y_pos + 1; j++, v++) {
				if(j >= 0 && tile_y_pos < tile_y) {
					// Ensure tile is within range

					for(int k = 0 ; k < points_per_tile ; k++){
						// Checking all points in current tile

						float x_point = tiles_x[position(u, v, k, 3, 3)]; //random_points_x[position(i, j, k, tile_x, tile_y)];
						float y_point = random_points_y[position(i, j, k, tile_x, tile_y)];
						float x_dist = (x - x_point) / intensity;
						float y_dist = (y - y_point) / intensity;

//						int distance = abs(x_dist) + abs(y_dist); // Manhattan distance
						int distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euclidean distance

						if(distance < shortest_norm_dist) {
							second_shortest_norm_dist = shortest_norm_dist;
							shortest_norm_dist = distance;
						} else if(distance < second_shortest_norm_dist) {
							second_shortest_norm_dist = distance;
						}
					}
				}
			}
		}
	}


	if(distance_order == 1) {
		result[position2D(x, y, height)] = shortest_norm_dist;
	} else {
		result[position2D(x, y, height)] = second_shortest_norm_dist;
	}
}


void WorleyNoise(const std::string outfile, const int width, const int height,
		         const int tile_size, const int points_per_tile, const float intensity, int seed, const int distance_order) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);
	assert(distance_order == 1 || distance_order == 2);

	if(seed == 0)
		seed = time(NULL);

	std::cout << "Creating Worley Noise with size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << ", distance order: " << distance_order << std::endl;

	// start timer
	double t = jbutil::gettime();

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	jbutil::randgen rand(seed);

	size_t random_points_size = tile_x * tile_y * points_per_tile * sizeof(int);
	int *random_points_x = (int *) malloc(random_points_size);
	int *random_points_y = (int *) malloc(random_points_size);

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int z = 0; z < points_per_tile; z++) {
				rand.advance();
				random_points_x[position(x, y, z, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				rand.advance();
				random_points_y[position(x, y, z, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);
			}
		}
	}

	int *d_result, *d_random_points_x, *d_random_points_y;

	size_t res_size = width * height * sizeof(int);

	gpuErrchk( cudaMalloc((void**) &d_result, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_x, random_points_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_y, random_points_size) );

	gpuErrchk( cudaMemcpy(d_random_points_x, random_points_x, random_points_size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_random_points_y, random_points_y, random_points_size, cudaMemcpyHostToDevice) );

	dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);
	dim3 blocks(32, 32);

    int sharedMemory = 9 * points_per_tile * sizeof(int);
//    normDistanceFromNearestPointSharedMem<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, distance_order, d_result);
	normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, distance_order, d_result);
//    gpuErrchk( cudaPeekAtLastError() );
//    gpuErrchk( cudaDeviceSynchronize() );

	int *result = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result, d_result, res_size, cudaMemcpyDeviceToHost) );

   for(int x = 0; x < width; x++) {
	   for(int y = 0; y < height; y++) {
		   image_out(0, y, x) = result[position2D(x, y, height)];
	   }
	}

	gpuErrchk( cudaFree(d_result) );
	gpuErrchk( cudaFree(d_random_points_x) );
	gpuErrchk( cudaFree(d_random_points_y) );

	free(result);

	// stop timer
	t = jbutil::gettime() - t;
	// save image
	std::ofstream file_out(outfile.c_str());
	image_out.save(file_out);
	// show time taken
	std::cerr << "Time taken: " << t << "s" << std::endl;
}


void Performance(const std::string outfile, const int width, const int height,
		         const int tile_size, const int points_per_tile, const float intensity, int seed, const int distance_order) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);
	assert(distance_order == 1 || distance_order == 2);

	if(seed == 0)
		seed = time(NULL);

	std::cout << "Creating Worley Noise with size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << ", distance order: " << distance_order << std::endl;

	// start timer
	double t = jbutil::gettime();

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	jbutil::randgen rand(seed);

	size_t random_points_size = tile_x * tile_y * points_per_tile * sizeof(int);
	int *random_points_x = (int *) malloc(random_points_size);
	int *random_points_y = (int *) malloc(random_points_size);

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int z = 0; z < points_per_tile; z++) {
				rand.advance();
				random_points_x[position(x, y, z, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				rand.advance();
				random_points_y[position(x, y, z, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);
			}
		}
	}

	int *d_result, *d_random_points_x, *d_random_points_y;

	size_t res_size = width * height * sizeof(int);

	gpuErrchk( cudaMalloc((void**) &d_result, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_x, random_points_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_y, random_points_size) );

	gpuErrchk( cudaMemcpy(d_random_points_x, random_points_x, random_points_size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_random_points_y, random_points_y, random_points_size, cudaMemcpyHostToDevice) );

	dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);
	dim3 blocks(32, 32);

    int sharedMemory = 9 * points_per_tile * sizeof(int);


    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    int c = 0;
    while(++c < 100)
    	normDistanceFromNearestPointSharedMem<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, distance_order, d_result);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

	std::cout << "Time using shared mem :" << time << "ms\n";

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    c = 0;
    while(++c < 100)
    	normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, distance_order, d_result);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

	std::cout << "Time w/o shared mem :" << time << "ms\n";


	int *result = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result, d_result, res_size, cudaMemcpyDeviceToHost) );

   for(int x = 0; x < width; x++) {
	   for(int y = 0; y < height; y++) {
		   image_out(0, y, x) = result[position2D(x, y, height)];
	   }
	}

	gpuErrchk( cudaFree(d_result) );
	gpuErrchk( cudaFree(d_random_points_x) );
	gpuErrchk( cudaFree(d_random_points_y) );

	free(result);

	// stop timer
	t = jbutil::gettime() - t;
	// save image
	std::ofstream file_out(outfile.c_str());
	image_out.save(file_out);
	// show time taken
	std::cerr << "Time taken: " << t << "s" << std::endl;
}
void printHelp(char *input) {
	std::cout << "Worley Noise\n"
			  << "Usage: "<< input << " [FILE] [OPTIONS] \n"
			  << "File should end with .pgm. List of options is shown below:\n"
			  << " -w, --width         width of image\n"
			  << " -b, --breath        breadth of image\n"
			  << " -t, --tilesize      image split in square tiles of the chosen size\n"
			  << " -p, --pptile        random pixels per tile\n"
			  << " -i, --intensity     intensity\n"
			  << " -s, --seed          preconfigure seed. If not configures, a random seed is chosen\n"
			  << " -d, --distanceorder distance order set to either 1 or 2; referring to first and second order distance respectively\n"
			  << " -h, --help          display options\n";
}

// Main program entry point

int main (int argc, char **argv) {
	// Default
	char *out = "out.pgm";
	int width = 2000;
	int height = 2500;
	int tile_size = 512;
	int points_per_tile = 50;
	float intensity = 1;
	int seed = 32234;
	int distance_order = 1;


	int index;
	int c;

    static struct option long_options[] = {
        {"width",        required_argument, 0,  'w' },
        {"breadth",      required_argument, 0,  'b' },
        {"tilesize",     required_argument, 0,  't' },
        {"pptile",       required_argument, 0,  'p' },
        {"intensity",    required_argument, 0,  'i' },
        {"seed",         required_argument, 0,  's' },
        {"distanceorder",required_argument, 0,  'd' },
        {"help",         no_argument,       0,  'h' },
        {0,           0,                 0,  0   }
    };

    int long_index =0;
    while ((c = getopt_long(argc, argv, "w:hb:t:p:i:s:d:em",
                   long_options, &long_index )) != -1) {
    	switch (c) {
			case 'h':
				printHelp(argv[0]);
				return 0;
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 'b':
				height = atoi(optarg);
				break;
			case 't':
				tile_size = atoi(optarg);
				break;
			case 'p':
				points_per_tile = atoi(optarg);
				break;
			case 'i':
				intensity = atof(optarg);
				break;
			case 's':
				seed = atoi(optarg);
				break;
			case 'd':
				distance_order = atoi(optarg);
				break;
			case '?':
				if (optopt == 'w' || optopt == 'b' || optopt == 't' || optopt == 'p' || optopt == 'i' || optopt == 's' || optopt == 'd')
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else
				if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr,
							"Unknown option character `\\x%x'.\n",
							optopt);
				return 1;
			default:
				abort ();
		}
	}


	for (index = optind; index < argc; index++) {
		out = argv[index];
	}

//	for(int x = 0; x < 3; x++) {
//		for(int y = 0; y < 3; y++) {
//			for(int z = 0; z < 3; z++) {
//				std::cout << x << ", " << y << " " << position(x,y,z, 3, 3) << "\n";
//			}
//		}
//	}

//	Performance(out, width, height, tile_size, points_per_tile, intensity, seed, distance_order);
	WorleyNoise(out, width, height, tile_size, points_per_tile, intensity, seed, distance_order);

	return 0;
}
