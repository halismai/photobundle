#include "imgproc.h"
#include <immintrin.h>

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

void imgradient(const uint8_t* src_ptr, const ImageSize& im_size, float* Ix_ptr, float* Iy_ptr)
{
  auto rows = im_size.rows;
  auto cols = im_size.cols;

  memset(Ix_ptr, 0.0, sizeof(float) * cols);
  memset(Iy_ptr, 0.0, sizeof(float) * cols);

  for(int y = 1; y < rows - 1; ++y)
  {
    auto srow = src_ptr + y*cols;
    auto Ix_row = Ix_ptr + y*cols;
    auto Iy_row = Iy_ptr + y*cols;

    Ix_row[0] = 0.0f;
    Iy_row[0] = 0.0f;
    for(int x = 1; x < cols - 1; ++x)
    {
      Ix_row[x] = 0.5f * ((float) srow[x+1] - (float) srow[x-1]);
      Iy_row[x] = 0.5f * ((float) srow[x+cols] - (float) srow[x-cols]);
    }
    Ix_row[cols-1] = 0.0f;
    Iy_row[cols-1] = 0.0f;
  }

  memset(Ix_ptr + (rows-1)*cols, 0.0f, sizeof(float) * cols);
  memset(Iy_ptr + (rows-1)*cols, 0.0f, sizeof(float) * cols);
}

#if defined(WITH_TBB)
struct ImageGradientFunc
{
  ImageGradientFunc(const uint8_t* src_ptr, int cols, float* Ix_ptr, float* Iy_ptr)
      : _src(src_ptr), _cols(cols), _Ix(Ix_ptr), _Iy(Iy_ptr) {}

  void operator()(const tbb::blocked_range<int>& range) const
  {
    for(int y = range.begin(); y != range.end(); ++y)
    {
      auto srow = _src + y*_cols;
      auto Ix_row = _Ix + y*_cols;
      auto Iy_row = _Iy + y*_cols;

      Ix_row[0] = 0.0f;
      Iy_row[0] = 0.0f;
      for(int x = 1; x < _cols - 1; ++x)
      {
        Ix_row[x] = 0.5f * ((float) srow[x+1] - (float) srow[x-1]);
        Iy_row[x] = 0.5f * ((float) srow[x+_cols] - (float) srow[x-_cols]);
      }
      Ix_row[_cols-1] = 0.0;
      Iy_row[_cols-1] = 0.0;
    }
  }

 private:
  const uint8_t* _src;
  int _cols;
  float* _Ix;
  float* _Iy;
}; // ImageGradientFunc
#endif


void imgradientParallel(const uint8_t* src_ptr, const ImageSize& im_size, float* Ix_ptr, float* Iy_ptr)
{
#if defined(WITH_TBB)
  auto rows = im_size.rows;
  auto cols = im_size.cols;

  memset(Ix_ptr, 0.0, sizeof(float) * cols);
  memset(Iy_ptr, 0.0, sizeof(float) * cols);

  tbb::parallel_for(tbb::blocked_range<int>(1, rows-1), ImageGradientFunc(src_ptr, cols, Ix_ptr, Iy_ptr));

  memset(Ix_ptr + (rows-1)*cols, 0.0f, sizeof(float) * cols);
  memset(Iy_ptr + (rows-1)*cols, 0.0f, sizeof(float) * cols);
#else
  imgradient(src_ptr, im_size, Ix_ptr, Iy_ptr);
#endif
}
