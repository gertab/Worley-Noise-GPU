//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "jbutil.h"

#define position(x, y, z, width, depth) (x + width * (y + depth * z))

// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

int normDistanceFromNearestPoint(int x, int y, int width, int height, int *random_points_x, int *random_points_y, int tile_size, unsigned int N) {

	int tile_x_pos = x / tile_size;
	int tile_y_pos = y / tile_size;

	int shortest_norm_dist = 255;

	if(x == 2016 && y == 0)
		std::cout << "c";
	int c = 0;
	for(int i = tile_x_pos - 1; i <= tile_x_pos + 1; i++) {
		if(i >= 0 && tile_x_pos < DIV_CEIL(width, tile_size)) {
			for(int j = tile_y_pos - 1; j <= tile_y_pos + 1; j++) {
				if(j >= 0 && tile_y_pos < DIV_CEIL(height, tile_size)) {

					c++;

					float x_point = random_points_x[position(i, j, 0, DIV_CEIL(width, tile_size), N)];
					float y_point = random_points_y[position(i, j, 0, DIV_CEIL(width, tile_size), N)];
					float x_dist = (x - x_point) / 2.0;
					float y_dist = (y - y_point) / 2.0;
					//
					////		int distance = abs(x_dist) + abs(y_dist); // Manhattan distance
					int distance = sqrt(x_dist * x_dist + y_dist * y_dist); // Euchlidian distance

					shortest_norm_dist = std::min(distance, shortest_norm_dist);
				}
			}
		}
	}

	if(c == 0)
		std::cout << c << std::endl;

	return shortest_norm_dist;
}

//        x_dist = (pixel_x - point_x) / (img_width / 4)
//        y_dist = (pixel_y - point_y) / (img_height / 4)
//        norm_dist = math.sqrt(x_dist ** 2 + y_dist ** 2)

template <class real>
void process(const std::string infile, const std::string outfile,
  const real R, const int a)
{
	// start timer
	double t = jbutil::gettime();

	int height = 3000, width = 2000;
	unsigned int N = 2;
	int seed = 70;
	unsigned int tile_size = 32;
	int tile_x = DIV_CEIL(width, tile_size);
	int tile_y = DIV_CEIL(height, tile_size);

	jbutil::randgen rand(seed);

	int *random_points_x = (int *) malloc(tile_x * tile_y * N * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * N * sizeof(int));

	for(int x = 0; x < tile_x; x++) {
		for(int y = 0; y < tile_y; y++) {
			for(int i = 0; i < N; i++) {
				rand.advance();
				random_points_x[position(x, y, i, tile_x, N)] = (int) rand.fval(0, width);
				rand.advance();
				random_points_y[position(x, y, i, tile_x, N)] = (int) rand.fval(0, height);
			}
		}
	}

	jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

	for(int i = 0; i < height; i++) {
	   for(int j = 0; j < width; j++) {
		   image_out(0, i, j) = normDistanceFromNearestPoint(j, i, width, height, random_points_x, random_points_y, tile_size, N)  % 255;
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

