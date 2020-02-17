//============================================================================
// Name        : WorleySerial.cpp
// Author      : Gerard Tabone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "shared/jbutil.h"


int normDistanceFromNearestPoint(int x, int y, float width, float height, std::vector<std::vector<int>> random_points) {

	int shortest_norm_dist = 100;

	for(unsigned int i = 0; i < random_points.size(); i++) {
		int x_point = random_points.at(i).at(0);
		int y_point = random_points.at(i).at(1);
		float x_dist = (x - x_point);
		float y_dist = (y - y_point);

		int distance = sqrt(x_dist * x_dist + y_dist * y_dist);

		shortest_norm_dist = std::min(distance, shortest_norm_dist);
	}

	return shortest_norm_dist;
}

//def _get_normalized_distance_from_nearest_point(pixel_x: int,
//                                                pixel_y: int,
//                                                img_width: int,
//                                                img_height: int,
//                                                random_points: list):
//    shortest_norm_dist = 1
//    for point_x, point_y in random_points:
//        x_dist = (pixel_x - point_x) / (img_width / 4)
//        y_dist = (pixel_y - point_y) / (img_height / 4)
//        norm_dist = math.sqrt(x_dist ** 2 + y_dist ** 2)
//
//        shortest_norm_dist = min(norm_dist, shortest_norm_dist)
//
//    return shortest_norm_dist

template <class real>
void process(const std::string infile, const std::string outfile,
      const real R, const int a)
   {
   // start timer
   double t = jbutil::gettime();


   int height = 1000, width = 1000;


	std::vector<std::vector<int>> random_points = std::vector<std::vector<int>>();


	jbutil::randgen rand(4);

	for(int i = 0; i <  100; i++) {
		rand.advance();

		int x = (int) rand.fval(0, width), y = (int) rand.fval(0, width);
		random_points.push_back({x, y});

	}

   // TODO: Implemented image resampling here
   jbutil::image<int> image_out = jbutil::image<int>(height, width, 1, 255);

   for(int i = 0; i < height; i++) {
	   for(int j = 0; j < width; j++) {
		   image_out(0, i, j) = normDistanceFromNearestPoint(j, i, width, height, random_points)  % 255;
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
	process<float> ("", "out", 0.1, 1);


}

