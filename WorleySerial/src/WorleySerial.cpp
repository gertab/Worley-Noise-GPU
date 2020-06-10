//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation
//============================================================================

#include "jbutil.h"
#include "WorleySerial.h"
#include "Tests.h"

// Creates Worley Noise depending on the options
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
	assert(tile_x > 0 && tile_y > 0);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	// Generate random points
	randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
			image_out(0, y, x) = normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);

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
	std::cerr << "Time taken: " << t << "s" << std::endl;
}

// Works the normalized distances from closest pixel (x, y) to the nearest point from (random_point_x, random_point_y)
// No shared memory used
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
				random_points_x[position3D(x, y, z, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				random_points_y[position3D(x, y, z, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);
			}
		}
	}
}

// Performance checking for Worley Noise
void PerformanceCheck(const int width, const int height,
		         const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse) {

	assert(intensity >= 1);
	assert(width > 0 && height > 0);
	assert(tile_size > 0 && points_per_tile > 0);

	if(seed == 0)
		seed = time(NULL); // Random seed

	std::cout << "Performance testing using the following configurations; size: " << width << "x" << height << ", tile size: "
			  << tile_size << "x" << tile_size << ", points per tile: " << points_per_tile << ", intensity: " << intensity
			  << ", seed: " << seed << std::endl;

	// Split space int tiles of size 'tile_size'
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_x > 0 && tile_y > 0);

	jbutil::randgen rand(seed);

	int *random_points_x, *random_points_y;

	// start timer w/o random point generation
	double t = jbutil::gettime();

	int count = 0;

	// Loop for at least 60s
	while((jbutil::gettime() - t) < 60) {
		count++;
		// Random points
		random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
		random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
		// Generate random points
		randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
				normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);
			}
		}

		free(random_points_x);
		free(random_points_y);
	}

	// stop timer
	t = jbutil::gettime() - t;
	// show time taken
	std::cout << "\n\n";
	std::cerr << "Ran " << count << " iterations in " << t << "s. Average time taken: " << t / count << " s" << std::endl;
}
