//============================================================================
// Name        : Tests.cpp
// Author      : Gerard Tabone
// Version     :
// Description : Tests for worley noise
//============================================================================

#include "Tests.h"
#include "WorleySerial.h"

void runTests() {
	std::cout << "Running test cases\n";
	test1();
	test2();

	testDIVCEIL();
	testPosition3D();
	std::cout << "End of tests\n";
}

void test1() {
	// Testing small dataset, with image 3 x 3

	// Default
	int width = 3;
	int height = 3;
	int tile_size = 32;
	int points_per_tile = 1;
	float intensity = 1;
	int seed = 123;

	int tile_x = DIV_CEIL(width, tile_size);
	assert(tile_x == 1);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_y == 1);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	// Ensuring constant random point: (2, 2)
	random_points_x[0] = 2;
	random_points_y[0] = 2;

	// Correct output
	int result[] = {0, 1, 2, 1, 1, 1, 2, 1, 0};
	int i = 0;
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
		   int p = normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);

		   assert(result[i] == p);
		   i++;
		}
	}

	free(random_points_x);
	free(random_points_y);
}

void test2() {
	// Testing larger dataset, with image 10 x 10

	// Default
	int width = 10;
	int height = 10;
	int tile_size = 32;
	int points_per_tile = 1;
	float intensity = 1;
	int seed = 768;

	int tile_x = DIV_CEIL(width, tile_size);
	assert(tile_x == 1);
	int tile_y = DIV_CEIL(height, tile_size);
	assert(tile_y == 1);

	jbutil::randgen rand(seed);

	// Random points
	int *random_points_x = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));
	int *random_points_y = (int *) malloc(tile_x * tile_y * points_per_tile * sizeof(int));

	randomPointGeneration(random_points_x, random_points_y, rand, tile_x, tile_y, tile_size, points_per_tile);

	// Ensuring constant random points;
	assert(random_points_x[0] == 22);
	assert(random_points_y[0] == 1);

	// Expected output
	int result[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 3, 3, 3, 4, 5, 5, 6, 7, 8, 9, 4, 4, 4, 5, 5, 6, 7, 8, 8, 9, 5, 5, 5, 5, 6, 7, 7, 8, 9, 10, 6, 6, 6, 6, 7, 7, 8, 9, 10, 10, 7, 7, 7, 7, 8, 8, 9, 9, 10, 11, 8, 8, 8, 8, 8, 9, 10, 10, 11, 12, 9, 9, 9, 9, 9, 10, 10, 11, 12, 12};
	int i = 0;
	for(int x = 0; x < width; x++) {
		for(int y = 0; y < height; y++) {
		   int p = normDistanceFromNearestPoint(x, y, width, height, random_points_x, random_points_y, tile_size, points_per_tile, intensity);

		   assert(result[i] == p);
		   i++;
		}
	}

	free(random_points_x);
	free(random_points_y);
}

void testDIVCEIL() {
	// Test division and ceiling function

	assert(DIV_CEIL(0, 5) == 0);
	assert(DIV_CEIL(1, 5) == 1);
	assert(DIV_CEIL(2, 5) == 1);
	assert(DIV_CEIL(3, 5) == 1);
	assert(DIV_CEIL(4, 5) == 1);
	assert(DIV_CEIL(5, 5) == 1);
	assert(DIV_CEIL(6, 5) == 2);
	assert(DIV_CEIL(7, 5) == 2);
	assert(DIV_CEIL(8, 5) == 2);
}

void testPosition3D() {
	int width = 2, height = 2;

	assert(position3D(0, 0, 0, width, height) == 0);
	assert(position3D(1, 0, 0, width, height) == 1);
	assert(position3D(0, 1, 0, width, height) == 2);
	assert(position3D(1, 1, 0, width, height) == 3);
	assert(position3D(0, 0, 1, width, height) == 4);
	assert(position3D(1, 0, 1, width, height) == 5);
	assert(position3D(0, 1, 1, width, height) == 6);
	assert(position3D(1, 1, 1, width, height) == 7);
}

