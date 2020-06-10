//============================================================================
// Name        : WorleySerial.h
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation header
//============================================================================

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include "jbutil.h"

// Convert 3D position to the corresponding flattened 1D array position
#define position3D(x, y, z, WIDTH, HEIGHT) (HEIGHT*WIDTH*z + WIDTH*y + x)

// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

void printHelp(char *input);
void randomPointGeneration(int *random_points_x, int* random_points_y, jbutil::randgen rand, int tile_x, int tile_y, int tile_size, int points_per_tile);
int normDistanceFromNearestPoint(int x, int y, int width, int height, int *random_points_x, int *random_points_y, int tile_size, int points_per_tile, float intensity);
void WorleyNoise(const std::string outfile, const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse);
void printHelp(char *input);
void PerformanceCheck(const int width, const int height, const int tile_size, const int points_per_tile, const float intensity, int seed, const bool reverse);
