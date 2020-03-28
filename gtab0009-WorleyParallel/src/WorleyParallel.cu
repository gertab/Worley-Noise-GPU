//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation
//============================================================================


#include "jbutil.h"
#include "WorleyParallel.h"
//#include "Tests.h"

__host__ __device__
void print3DMatrix(int *data, int width, int height, int depth) {

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
__global__ void normDistanceFromNearestPointSharedMemory(int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity, int *result) {
	assert(tile_size > 0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);

//	Load to shared memory
	extern __shared__ int s[];
	int *tiles_x = s;
	int *tiles_y = (int*) &tiles_x[9 * points_per_tile];

	// Each thread in a block has a different index;
	int indexInBlock = threadIdx.x + blockDim.x * threadIdx.y;

	assert(blockDim.x * blockDim.y >= 9 * points_per_tile);

	if(indexInBlock < (9 * points_per_tile)) {
		int tileToGet = indexInBlock / points_per_tile;
		int shared_memory_z = indexInBlock % points_per_tile;

		// shared_memory_x can have value in range [0, 2]. Same for shared_memory_y
		int shared_memory_x = tileToGet % 3;
		int shared_memory_y = tileToGet / 3;

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
			tiles_x[position_shared_memory_1D] = 0;
			tiles_y[position_shared_memory_1D] = 0;
		}
	}

    if(x >= width || y >= height) {
    	// x and y  bigger than the limits are still used to load data in shared memory, since that if the tiles (in the right/bottom border)
    	// are bigger than the block size, they would still be needed
    	return;
    }

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
				int y_point = tiles_y[position3D(i, j, k, 3, 3)];

				if(!(x_point == 0 && y_point == 0)) {
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

	result[position3D(x, y, 0, width, height)] = shortest_norm_dist;
}


// Creates Worley Noise according to the options
void WorleyNoise(const std::string outfile, const int width, const int height,
		         const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);

	if(seed == 0)
		seed = time(NULL); // Random seed

	std::cout << "Creating Worley Noise using the GPU, with size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << std::endl;

	// start timer
	double t = jbutil::gettime();

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);

	jbutil::randgen rand(seed);

	// Random points
	size_t random_points_size = tile_x * tile_y * points_per_tile * sizeof(int);
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	// Generate random points
	randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	int *d_result, *d_random_points_x, *d_random_points_y;

	size_t res_size = width * height * sizeof(int);

	// Allocating memory on device
	gpuErrchk( cudaMalloc((void**) &d_result, res_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_x, random_points_size) );
	gpuErrchk( cudaMalloc((void**) &d_random_points_y, random_points_size) );

	// Copying data to device
	gpuErrchk( cudaMemcpy(d_random_points_x, random_points_x, random_points_size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_random_points_y, random_points_y, random_points_size, cudaMemcpyHostToDevice) );

	dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);
	dim3 blocks(32, 32);

//	normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);
    int sharedMemory = 2 * 9 * points_per_tile * sizeof(int);
    normDistanceFromNearestPointSharedMemory<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy result back to host from device
	int *result = (int *) malloc(res_size);
	gpuErrchk( cudaMemcpy(result, d_result, res_size, cudaMemcpyDeviceToHost) );


	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			image_out(0, y, x) = result[position3D(x, y, 0, width, height)]; // todo change to 2D

			if(reverse) {
				// Reverse image: white -> black, black -> white
				image_out(0, y, x) = 255 - image_out(0, y, x);
			}
		}
	}

	free(random_points_x);
	free(random_points_y);

	// stop timer
	t = jbutil::gettime() - t;

	// save image
	std::ofstream file_out(outfile.c_str());
	image_out.save(file_out);
	// show time taken
	std::cerr << "Total time taken: " << t << "s" << std::endl;
}


//// Performance checking for Worley Noise
//void PerformanceCheck(const int width, const int height,
//		         const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse) {
//
//	assert(intensity >= 1);
//	assert(width > 0 && height > 0);
//	assert(tile_size > 0 && points_per_tile > 0);
//
//	if(seed == 0)
//		seed = time(NULL); // Random seed
//
//	std::cout << "Performance testing using the following configurations; size: " << width << "x" << height << ", tile size: "
//			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
//			  << ", seed: " << seed << std::endl;
//
//	// Split space int tiles of size 'tile_size'
//	int tile_x = DIV_CEIL(width, tile_size);
//	int tile_y = DIV_CEIL(height, tile_size);
//	assert(tile_x > 0 && tile_y > 0);
//
//	jbutil::randgen rand(seed);
//
//	// Random points
//	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
//	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
//	// Generate random points
//	randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);
//
//	// start timer w/o random point generation
//	double t = jbutil::gettime();
//
//	int count = 0;
//
//	// todo do warmup
//	// Loop for at least 60s
//	while((jbutil::gettime() - t) < 60) {
//		count++;
//
//		for(int x = 0; x < width; x++) {
//			for(int y = 0; y < height; y++) {
//				normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);
//			}
//		}
//	}
//
//	free(random_points_x);
//	free(random_points_y);
//
//	// stop timer
//	t = jbutil::gettime() - t;
//	// show time taken
//	std::cout << "\n\n";
//	std::cerr << "Ran " << count << " iterations in " << t << "s. Average time taken: " << t / count << "s" << std::endl;
//}

void printHelp(char *input) {
	std::cout << "Worley Noise\n"
			  << "\tUsage: "<< input << " [FILE] [OPTIONS] \n\n"
			  << "File should end with .pgm. List of options is shown below:\n"
			  << " -w, --width         width of image\n"
			  << " -b, --breath        breadth of image\n"
			  << " -t, --tilesize      image split in square tiles of the chosen size\n"
			  << " -p, --pptile        random pixels per tile\n"
			  << " -i, --intensity     intensity\n"
			  << " -s, --seed          preconfigure seed. If not configures, a random seed is chosen\n"
			  << " -r, --reverse       the colours of the image will be inverted\n"
			  << "     --performance   used to time worley noise, without outputting anything\n"
			  << " -h, --help          display options\n"
			  << "\nExample commands:\n"
			  << "./WorleySerial out.pgm --width 2500 --breadth 1500 --seed 34534 --pptile 5 --intensity 1.5\n"
			  << "./WorleySerial out.pgm -w 500 -b 500 -p 5 -t 256\n"
			  << "./WorleySerial out.pgm -p 5 --reverse\n";
}

// Main program entry point
int main (int argc, char **argv) {
#ifdef RUNTESTS
	// Run test cases in Debug mode
	runTests();

#else
	// Default
	char const *out = "out.pgm";
//	int width = 2000;
//	int height = 2500;
//	int tile_size = 512;
//	int points_per_tile = 5;
//	float intensity = 1;
//	int seed = 0;
//	bool inverse = false;
//	bool performance = false;
	int width = 3500;
	int height = 3500;
	int tile_size = 512;
	int points_per_tile = 128;
	float intensity = 1.5;
	int seed = 782346;
	bool inverse = false;
	bool performance = false;

	int index;
	int c;

    static struct option long_options[] = {
        {"width",        required_argument, 0,  'w' },
        {"breadth",      required_argument, 0,  'b' },
        {"tilesize",     required_argument, 0,  't' },
        {"pptile",       required_argument, 0,  'p' },
        {"intensity",    required_argument, 0,  'i' },
        {"seed",         required_argument, 0,  's' },
        {"reverse",      no_argument, 		0,  'r' },
        {"help",         no_argument,       0,  'h' },
        {"performance",  no_argument,       0,  'k' },
        {0,           	 0,                 0,  0   }
    };

    int long_index =0;
    while ((c = getopt_long(argc, argv, "w:b:t:p:i:s:d:rhk",
                   long_options, &long_index )) != -1) {
    	switch (c) {
			case 'h':
			{
				printHelp(argv[0]);
				return 0;
				break;
			}
			case 'w':
			{
				int tmp = atoi(optarg);

				if(tmp > 0) {
					width = tmp;
				} else {
					std::cout << "Width must be > 0\n";
				}
				break;
			}
			case 'b':
			{
				int tmp = atoi(optarg);

				if(tmp > 0) {
					height = tmp;
				} else {
					std::cout << "height must be > 0\n";
				}
				break;
			}
			case 't':
			{
				int tmp = atoi(optarg);

				if(tmp > 0) {
					tile_size = tmp;
				} else {
					std::cout << "tile size must be > 0\n";
				}
				break;
			}
			case 'p':
			{
				int tmp = atoi(optarg);

				if(tmp > 0) {
					points_per_tile = tmp;
				} else {
					std::cout << "points per tile must be > 0\n";
				}
				break;
			}
			case 'i':
			{
				float tmp = atof(optarg);

				if(tmp >= 1) {
					intensity = tmp;
				} else {
					std::cout << "intensity must be >= 1\n";
				}
				break;
			}
			case 's':
				seed = atoi(optarg);
				break;
			case 'r':
				inverse = true;
				break;
			case 'k':
				performance = true;
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

	if(!performance) {
		WorleyNoise(out, width, height, tile_size, points_per_tile, intensity, seed, inverse);
	} else {
		// No outputs
//		PerformanceCheck(width, height, tile_size, points_per_tile, intensity, seed, inverse);
	}
#endif
	return 0;
}
