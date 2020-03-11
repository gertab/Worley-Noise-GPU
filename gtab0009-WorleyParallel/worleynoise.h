#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Convert 3D position to the corresponding flattened 1D array position
#define position(x, y, z, WIDTH, HEIGHT) (HEIGHT*WIDTH*z + WIDTH*y + x)
#define position2D(x, y, HEIGHT) ( x * HEIGHT + y)


// ceil( x / y )
#define DIV_CEIL(x, y) ((x + y - 1) / y)

