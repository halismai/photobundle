#include "imgproc.h"
#include <immintrin.h>

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

/**
 * comptue the image gradient for a row of pixels in the image
 */
template <typename TSrc, typename TDst> inline
void imgradient_row(const TSrc* srow, int cols, TDst* Ix_row, TDst* Iy_row)
{
  static_assert( std::is_signed<TDst>::value, "imgradient destination type must be signed" );

  constexpr auto S = imgradient_scale<TDst>();

  Ix_row[0] = TDst(0);
  Iy_row[0] = TDst(0);

  for(int x = 1; x < cols - 1; ++x) {
    Ix_row[x] = S * (TDst(srow[x+1]) - TDst(srow[x-1]));
    Iy_row[x] = S * (TDst(srow[x+cols]) - TDst(srow[x-cols]));
  }

  Ix_row[cols-1] = TDst(0);
  Iy_row[cols-1] = TDst(0);
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

      imgradient_row(srow, _cols, Ix_row, Iy_row);
    }
  }

 private:
  const uint8_t* _src;
  int _cols;
  float* _Ix;
  float* _Iy;
}; // ImageGradientFunc
#endif


void imgradient(const uint8_t* src_ptr, const ImageSize& im_size, float* Ix_ptr, float* Iy_ptr)
{
  auto rows = im_size.rows;
  auto cols = im_size.cols;

  memset(Ix_ptr, 0.0, sizeof(float) * cols);
  memset(Iy_ptr, 0.0, sizeof(float) * cols);

#if defined(WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(1, rows-1), ImageGradientFunc(src_ptr, cols, Ix_ptr, Iy_ptr));
#else
  for(int y = 1; y < rows - 1; ++y) {
    auto srow = src_ptr + y*cols;
    auto Ix_row = Ix_ptr + y*cols;
    auto Iy_row = Iy_ptr + y*cols;
    imgradient_row(srow, cols, Ix_row, Iy_row);
  }
#endif
  memset(Ix_ptr + (rows-1)*cols, 0.0f, sizeof(float) * cols);
  memset(Iy_ptr + (rows-1)*cols, 0.0f, sizeof(float) * cols);
}

