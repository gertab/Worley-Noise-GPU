//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "jbutil.h"


int normDistanceFromNearestPoint(int x, int y, float width, float height, int *random_points, unsigned int N) {

int shortest_norm_dist = 255;

for(unsigned int i = 0; i < N; i++) {
	float x_point = random_points[i * 2];
	float y_point = random_points[i * 2 + 1];
	float x_dist = (x - x_point) / 2.0;
	float y_dist = (y - y_point) / 2.0;

	int distance = sqrt(x_dist * x_dist + y_dist * y_dist);

	shortest_norm_dist = std::min(distance, shortest_norm_dist);
}

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


	int height = 400, width = 800;
	unsigned int N = 20;
	int seed = 5;
	jbutil::randgen rand(seed);

	int random_points[N][2];


	for(unsigned int i = 0; i <  N; i++) {
		rand.advance();
		random_points[i][0] = (int) rand.fval(0, width);
		rand.advance();
		random_points[i][1] = (int) rand.fval(0, height);
	}

	jbutil::image<int> image_out = jbutil::image<int>(width, height, 1, 255);

	for(int i = 0; i < width; i++) {
	   for(int j = 0; j < height; j++) {
		   image_out(0, i, j) = normDistanceFromNearestPoint(i, j, width, height, (int *) random_points, N)  % 255;
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

