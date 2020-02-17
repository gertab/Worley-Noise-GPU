///*!
// * \file
// * \brief   Lab 2 - SIMD Programming.
// * \author  Johann Briffa
// *
// * Template for the solution to Lab 2 practical exercise on image resampling
// * using the Lanczos filter.
// *
// * \section svn Version Control
// */
//
//#include jbutil.h"
//
//// Resample the image using Lanczos filter
//
//template <class real>
//void process(const std::string infile, const std::string outfile,
//      const real R, const int a)
//   {
//   // load image
//   jbutil::image<int> image_in;
//   std::ifstream file_in(infile.c_str());
//   image_in.load(file_in);
//   // start timer
//   double t = jbutil::gettime();
//
//   // TODO: Implemented image resampling here
//   jbutil::image<int> image_out = image_in;
//
//   // stop timer
//   t = jbutil::gettime() - t;
//   // save image
//   std::ofstream file_out(outfile.c_str());
//   image_out.save(file_out);
//   // show time taken
//   std::cerr << "Time taken: " << t << "s" << std::endl;
//   }
//
//// Main program entry point
//
//int main(int argc, char *argv[])
//   {
//   std::cerr << "Lab 2: Image resampling with Lanczos filter" << std::endl;
//   if (argc != 5)
//      {
//      std::cerr << "Usage: " << argv[0]
//            << " <infile> <outfile> <scale-factor> <limit>" << std::endl;
//      exit(1);
//      }
//   process<float> (argv[1], argv[2], atof(argv[3]), atoi(argv[4]));
//   }
