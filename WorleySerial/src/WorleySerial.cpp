//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <time.h>
#include "jbutil.h"


#define position(x, y, z, WIDTH, HEIGHT) (HEIGHT*WIDTH*z + WIDTH*y + x)

// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

int normDistanceFromNearestPoint(int x, int y, int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity) {

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	// 0   = black
	// 255 = white
	int shortest_norm_dist = 255;

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
						//
//								int distance = abs(x_dist) + abs(y_dist); // Manhattan distance
						int distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euclidean distance

			            shortest_norm_dist = std::min(distance, shortest_norm_dist);
					}
				}
			}
		}
	}

	return shortest_norm_dist;
}

void WorleyNoise(const std::string outfile, const int width, const int height,
		         const int tile_size, const int points_per_tile, const float intensity, int seed) {
	// start timer
	double t = jbutil::gettime();

	int N = points_per_tile;

	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	if(seed == 0)
		seed = time(NULL);

	jbutil::randgen rand(seed);

	int *random_points_x = (int *) malloc(tile_x * tile_y * N * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * N * sizeof(int));

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

//
//   for(int x = 0; x < width; x++) {
//	   for(int y = 0; y < height; y++) {
//		   image_out(0, y, x) = 255;
//	   }
//	}

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int z = 0; z < N; z++) {
				rand.advance();
				random_points_x[position(x, y, z, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				rand.advance();
				random_points_y[position(x, y, z, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);

//				int yy = random_points_y[position(x, y, i, tile_x, N)];
//				int xx = random_points_x[position(x, y, i, tile_x, N)];

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
			}
		}
	}

   for(int x = 0; x < width; x++) {
	   for(int y = 0; y < height; y++) {
		   image_out(0, y, x) = normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, N, intensity);
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

// Main program entry point

int main(int argc, char *argv[]) {
	std::cerr << "Worley Noise" << std::endl;
	if (argc != 5)
	{
		std::cerr << "Usage: " << argv[0]
		<< " <outfile> <width> <height> <tile size> <points per tile> <intensity> <seed>" << std::endl;


		// Default
		int width = 1000;
		int height = 1500;
		int tile_size = 300;
		int points_per_tile = 2;
		int intensity = 2;
		int seed = 234324;
		WorleyNoise("out.pgm", width, height, tile_size, points_per_tile, intensity, seed);
	} else {
		WorleyNoise(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atof(argv[6]), atoi(argv[7]));
	}
}

