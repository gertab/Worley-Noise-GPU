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

//	printf("%d, %d:::%d\n", x, y, shortest_norm_dist);

	result[position2D(x, y, height)] = shortest_norm_dist;

//	if(distance_order == 1) {
//		return shortest_norm_dist;
//	} else {
//		return second_shortest_norm_dist;
//	}
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

//
//   for(int x = 0; x < width; x++) {
//	   for(int y = 0; y < height; y++) {
//		   image_out(0, y, x) = 255;
//	   }
//	}

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

    dim3 threadsPerBlock(100, 100);
    dim3 threads(32, 32);

	normDistanceFromNearestPoint<<<threadsPerBlock, threads>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, distance_order, d_result);

	int *result = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result, d_result, res_size, cudaMemcpyDeviceToHost) );

   for(int x = 0; x < width; x++) {
	   for(int y = 0; y < height; y++) {
//		   std::cout << x << "  " << y << ", " << result[position2D(x, y, height)] << "\n";
		   image_out(0, y, x) = result[position2D(x, y, height)];
				   //normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity, distance_order);
	   }
	}

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
	int width = 1000;
	int height = 1500;
	int tile_size = 400;
	int points_per_tile = 2;
	float intensity = 2;
	int seed = 0;
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

	WorleyNoise(out, width, height, tile_size, points_per_tile, intensity, seed, distance_order);

	return 0;
}
