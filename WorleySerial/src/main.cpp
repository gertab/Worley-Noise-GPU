//============================================================================
// Name        : main.cpp
// Author      : Gerard Tabone
// Version     :
// Description : Worley noise simulation entry point
//============================================================================

#include "jbutil.h"
#include "WorleySerial.h"
#include "Tests.h"

// Main program entry point
int main (int argc, char **argv) {
#ifdef RUNTESTS
	// Run test cases in Debug mode
	runTests();

#else
	// Default
	char const *out = "out.pgm";
	int width = 2000;
	int height = 2500;
	int tile_size = 512;
	int points_per_tile = 5;
	float intensity = 1;
	int seed = 0;
	bool inverse = false;
	bool performance = false;

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
        {0,           	 0,                 0,  0   }
    };

    int long_index =0;
    while ((c = getopt_long(argc, argv, "w:b:t:p:i:s:d:rhk",
                   long_options, &long_index )) != -1) {
    	switch (c) {
			case 'h':
			{
				printHelp(argv[0]);
				return 0;
				break;
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

				if(tmp > 0) {
					points_per_tile = tmp;
				} else {
					std::cout << "points per tile must be > 0\n";
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

	if(!performance) {
		WorleyNoise(out, width, height, tile_size, points_per_tile, intensity, seed, inverse);
	} else {
		// No outputs
		PerformanceCheck(width, height, tile_size, points_per_tile, intensity, seed, inverse);
	}
#endif
	return 0;
}

void printHelp(char *input) {
	std::cout << "Worley Noise\n"
			  << "\tUsage: "<< input << " [FILE] [OPTIONS] \n\n"
			  << "File should end with .pgm. List of options is shown below:\n"
			  << " -w, --width         width of image\n"
			  << " -b, --breath        breadth of image\n"
			  << " -t, --tilesize      image split in square tiles of the chosen size\n"
			  << " -p, --pptile        random pixels per tile\n"
			  << " -i, --intensity     intensity\n"
			  << " -s, --seed          preconfigure seed. If not configures, a random seed is chosen\n"
			  << " -r, --reverse       the colours of the image will be inverted\n"
			  << "     --performance   used to time worley noise, without outputting anything\n"
			  << " -h, --help          display options\n"
			  << "\nExample commands:\n"
			  << "./WorleySerial out.pgm --width 2500 --breadth 1500 --seed 34534 --pptile 5 --intensity 1.5\n"
			  << "./WorleySerial out.pgm -w 500 -b 500 -p 5 -t 256\n"
			  << "./WorleySerial out.pgm -p 5 --reverse\n";
}
