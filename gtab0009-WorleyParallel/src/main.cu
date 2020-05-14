//============================================================================
// Name        : main.cu
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation
//============================================================================


#include "jbutil.h"
#include "WorleyParallel.h"
#include "Tests.h"
#include "main.h"
#include <curand.h>
#include <curand_kernel.h>

// Main program entry point
int main (int argc, char **argv) {

#ifdef RUNTESTS
	// Run test cases in Debug mode
	runTests();

#else
	// Default
//	char const *out = "out.pgm";
//	int width = 2000;
//	int height = 2500;
//	int tile_size = 512;
//	int points_per_tile = 5;
//	float intensity = 1;
//	int seed = 0;
//	bool inverse = false;
//	bool performance = false;
//	bool shared_memory = false;
//	bool fast_math = false;

	char const *out = "out.pgm";
	int width = 8000;
	int height = 8000;
	int tile_size = 512;
	int points_per_tile = 16;
	float intensity = 1;
	int seed = 234723;
	bool inverse = false;
	bool performance = true;
	bool shared_memory = false;
	bool fast_math = true;

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
        {"sharedmemory", no_argument,       0,  'z' },
        {"fastmath",  	 no_argument,       0,  'f' },
        {0,           	 0,                 0,  0   }
    };

    int long_index =0;
    while ((c = getopt_long(argc, argv, "w:b:t:p:i:s:d:rhkzf",
                   long_options, &long_index )) != -1) {
    	switch (c) {
			case 'h':
			{
				printHelp(argv[0]);
				return 0;
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

				if(tmp > 0 && tmp <= 113) {
					points_per_tile = tmp;
				} else {
					std::cout << "points per tile must be between 1 and 113 (both inclusive)\n";
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
			case 'f':
				fast_math = true;
				break;
			case 'z':
				shared_memory = true;
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

	if(performance) {
		// No outputs
		PerformanceCheck(width, height, tile_size, points_per_tile, intensity, seed, inverse, shared_memory, fast_math);
	} else {
		WorleyNoise(out, width, height, tile_size, points_per_tile, intensity, seed, inverse, shared_memory, fast_math);
	}
#endif
	return 0;
}


// Creates Worley Noise according to the options
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

	if(false) {
		// Generate random number on the host and transfer results to device

		int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
		int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
		// Generate random points
		randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

		// Copying data to device
		gpuErrchk( cudaMemcpy(d_random_points_x, random_points_x, random_points_size, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_random_points_y, random_points_y, random_points_size, cudaMemcpyHostToDevice) );

		free(random_points_x);
		free(random_points_y);
	} else {
		// Generate random number directly on device

		generateRandomPointOnDevice(d_random_points_x, d_random_points_y, seed, tile_x, tile_y, tile_size, points_per_tile);
	}

	dim3 grid(DIV_CEIL(width, 32), DIV_CEIL(height, 32));
	dim3 blocks(32, 32);

	if(shared_memory) {
		std::cout << "Using shared memory";
		if(fast_math) {
			std::cout << " and fast math";
		}
		std::cout << "\n";

	    int sharedMemory = 2 * 9 * points_per_tile * sizeof(int);
    	normDistanceFromNearestPointSharedMemory<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result, fast_math);
	} else  {
		std::cout << "Without using shared memory\n";
		normDistanceFromNearestPoint<<<grid, blocks>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);
	}

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

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

//	// Warmup
//	int sharedMemory = 2 * 9 * points_per_tile * sizeof(int);
//	normDistanceFromNearestPointSharedMemory<<<grid, blocks, sharedMemory>>>(width, height, d_random_points_x, d_random_points_y, tile_size, points_per_tile, intensity, d_result);

	// Loop for at least 60 seconds
	while((jbutil::gettime() - t) < 60) {
		count++;

		if(false) {
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
		} else {
			// Generate random number directly on device

			generateRandomPointOnDevice(d_random_points_x, d_random_points_y, seed, tile_x, tile_y, tile_size, points_per_tile);
		}

		dim3 grid(DIV_CEIL(width, 32), DIV_CEIL(height, 32));
		dim3 blocks(32, 32);

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
	std::cerr << "Ran " << count << " iterations in " << t << "s. Average time taken: " << t / count << "s" << std::endl;
}

void printHelp(char *input) {
	std::cout << "Worley Noise\n"
			  << "\tUsage: "<< input << " [FILE] [OPTIONS] \n\n"
			  << "File should end with .pgm. List of options is shown below:\n"
			  << " -w, --width         width of image\n"
			  << " -b, --breath        breadth of image\n"
			  << " -t, --tilesize      image split in square tiles of the chosen size\n"
			  << " -p, --pptile        random pixels per tile. Nede to be between 1 and 113 (both inclusive)\n"
			  << " -i, --intensity     intensity\n"
			  << " -s, --seed          preconfigure seed. If not configures, a random seed is chosen\n"
			  << " -r, --reverse       the colours of the image will be inverted\n"
			  << " -f, --fastmath      uses faster but less accurate math operations\n"
			  << "     --performance   used to time worley noise, without outputting anything\n"
			  << "     --sharedmemory  use GPU version with shared memory\n"
			  << " -h, --help          display options\n"
			  << "\nExample commands:\n"
			  << "./WorleySerial out.pgm --width 2500 --breadth 1500 --seed 34534 --pptile 5 --intensity 1.5 --sharedmemory\n"
			  << "./WorleySerial out.pgm -w 500 -b 500 -p 5 -t 256 --sharedmemory\n"
			  << "./WorleySerial out.pgm -p 5 --reverse\n";
}
