//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


#include "jbutil.h"
#include "WorleySerial.h"
#include "Tests.h"

#define RUNTESTS

// Fills random_points_x and random_points_y with random numbers
// random_points_x and random_points_y should have enough space to be filled with (tile_x * tile_y * points_per_tile) random numbers
void randomPointGeneration(int *random_points_x, int* random_points_y, jbutil::randgen rand, int tile_x, int tile_y, int tile_size, int points_per_tile) {
	assert(random_points_x != nullptr && random_points_y != nullptr);
	assert(tile_x > 0 && tile_y > 0);
	assert(tile_size > 0);
	assert(points_per_tile > 0);

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int z = 0; z < points_per_tile; z++) {
				rand.advance();
				random_points_x[position3D(x, y, z, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				rand.advance();
				random_points_y[position3D(x, y, z, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);
			}
		}
	}

}

// Works the normalized distances from pixel (x, y) to the nearest point from (random_point_x, random_point_y)
int normDistanceFromNearestPoint(int x, int y, int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity) {

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

//						int distance = abs(x_dist) + abs(y_dist); // Manhattan distance
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

void WorleyNoise(const std::string outfile, const int width, const int height,
		         const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);

	if(seed == 0)
		seed = time(NULL); // Random seed

	std::cout << "Creating Worley Noise with size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << std::endl;

	// start timer
	double t = jbutil::gettime();

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

//
//   for(int x = 0; x < width; x++) {
//	   for(int y = 0; y < height; y++) {
//		   image_out(0, y, x) = 255;
//	   }
//	}
//	for(int x = 0; x < tile_x; x++) {
//		for(int y = 0; y < tile_y; y++) {
//			for(int z = 0; z < points_per_tile; z++) {
//				int yy = random_points_y[position3D(x, y, i, tile_x, points_per_tile)];
//				int xx = random_points_x[position3D(x, y, i, tile_x, points_per_tile)];
//
//				if(xx < width && yy < height) {
//					image_out(0, yy + 1, xx) = 0;
//					image_out(0, yy, xx) = 0;
//					image_out(0, yy - 1, xx) = 0;
//					image_out(0, yy + 1, xx - 1) = 0;
//					image_out(0, yy, xx - 1) = 0;
//					image_out(0, yy - 1, xx - 1) = 0;
//					image_out(0, yy + 1, xx + 1) = 0;
//					image_out(0, yy, xx + 1) = 0;
//					image_out(0, yy - 1, xx + 1) = 0;
//				}
//			}
//		}
//	}

   for(int x = 0; x < width; x++) {
	   for(int y = 0; y < height; y++) {
		   image_out(0, y, x) = normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);

		   if(reverse) {
			   // Revere image: white -> black, black -> white
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
			  << " -r, --reverse       the colours of the image will be inverted\n"
			  << " -h, --help          display options\n";
}

// Main program entry point
int main (int argc, char **argv) {
	// TODO: Do simple 3x3 or 10x10 output
#ifndef RUNTESTS

	// Default
	char *out = "out.pgm";
	int width = 2000;
	int height = 2500;
	int tile_size = 512;
	int points_per_tile = 5;
	float intensity = 1;
	int seed = 0;
	bool inverse = false;

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
        {0,           	 0,                 0,  0   }
    };

    int long_index =0;
    while ((c = getopt_long(argc, argv, "w:b:t:p:i:s:d:rh",
                   long_options, &long_index )) != -1) {
    	switch (c) {
			case 'h':
				printHelp(argv[0]);
				return 0;
				break;
			case 'x':
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
			case 'r':
				inverse = true;
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

	WorleyNoise(out, width, height, tile_size, points_per_tile, intensity, seed, inverse);
#else
	// Run test cases
	runTests();
#endif
	return 0;
}
