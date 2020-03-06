//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "jbutil.h"

#define position(x, y, z, WIDTH, HEIGHT) (HEIGHT*WIDTH*z + WIDTH*y + x)

// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

int normDistanceFromNearestPoint(int x, int y, int width, int height, int *random_points_x, int *random_points_y, int tile_size, int N) {

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int shortest_norm_dist = 255;

	for(int i = tile_x_pos - 1; i <= tile_x_pos + 1; i++) {
		if(i >= 0 && tile_x_pos < DIV_CEIL(width, tile_size)) {
			for(int j = tile_y_pos - 1; j <= tile_y_pos + 1; j++) {
				if(j >= 0 && tile_y_pos < DIV_CEIL(height, tile_size)) {

					for(int k = 0 ; k < N ; k++){

						float x_point = random_points_x[position(i, j, k, DIV_CEIL(width, tile_size), DIV_CEIL(height, tile_size))];
						float y_point = random_points_y[position(i, j, k, DIV_CEIL(width, tile_size), DIV_CEIL(height, tile_size))];
						float x_dist = (x - x_point) / 1.5;
						float y_dist = (y - y_point) / 1.5;
						//
						////		int distance = abs(x_dist) + abs(y_dist); // Manhattan distance
						int distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euchlidian distance

			            shortest_norm_dist = std::min(distance, shortest_norm_dist);
					}
				}
			}
		}
	}

	return shortest_norm_dist;
}

template <class real>
void process(const std::string infile, const std::string outfile,
  const real R, const int a)
{
	// start timer
	double t = jbutil::gettime();

	int height = 2600, width = 1200;
	int N = 4;
	int seed = 878;
	int tile_size = 200;
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	jbutil::randgen rand(seed);

	int *random_points_x = (int *) malloc(tile_x * tile_y * N * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * N * sizeof(int));

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);


   for(int x = 0; x < width; x++) {
	   for(int y = 0; y < height; y++) {
		   image_out(0, y, x) = 255;
	   }
	}

//	for(int x = 0; x < tile_x; x++) {
//		for(int y = 0; y < tile_y; y++) {
//			for(int i = 0; i < N; i++) {
//				std::cout << "[" << x << ", " << y << ", " << i << "] : " << position(x, y, i, tile_x, tile_y) << std::endl;
//			}
//		}
//	}

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int i = 0; i < N; i++) {
				rand.advance();
				random_points_x[position(x, y, i, tile_x, tile_y)] = (int) rand.fval(x * tile_size, (x + 1) * tile_size);
				rand.advance();
				random_points_y[position(x, y, i, tile_x, tile_y)] = (int) rand.fval(y * tile_size, (y + 1) * tile_size);

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
		   int val =  normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, N);
		   image_out(0, y, x) = val;
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

int main(int argc, char *argv[])
{
	//	std::cerr << "Lab 2: Image resampling with Lanczos filter" << std::endl;
	//	if (argc != 5)
	//	{
	//		std::cerr << "Usage: " << argv[0]
	//		<< " <infile> <outfile> <scale-factor> <limit>" << std::endl;
	//		exit(1);
	//	}
	//	process<float> (argv[1], argv[2], atof(argv[3]), atoi(argv[4]));
	process<float> ("", "out.pgm", 0.1, 1);


}

