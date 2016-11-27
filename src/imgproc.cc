#include "imgproc.h"
#include "debug.h"
#include "v128.h"

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

/**
 * comptue the image gradient for a row of pixels in the image
 */
template <typename TSrc, typename TDst> FORCE_INLINE
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

using bpvo::v128;

static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);


/**
 * computes the Census Transform for 16 pixels at once
 */
static FORCE_INLINE void censusOp(const uint8_t* src, int stride, uint8_t* dst)
{

#define C_OP >=
  const v128 c(src);
  _mm_storeu_si128((__m128i*) dst,
                   ((v128(src - stride - 1) C_OP c) & K0x01) |
                   ((v128(src - stride    ) C_OP c) & K0x02) |
                   ((v128(src - stride + 1) C_OP c) & K0x04) |
                   ((v128(src          - 1) C_OP c) & K0x08) |
                   ((v128(src          + 1) C_OP c) & K0x10) |
                   ((v128(src + stride - 1) C_OP c) & K0x20) |
                   ((v128(src + stride    ) C_OP c) & K0x40) |
                   ((v128(src + stride + 1) C_OP c) & K0x80));
#undef C_OP
}

static FORCE_INLINE void census_row(const uint8_t* srow, int cols, uint8_t* drow)
{
  drow[0] = 0;

  const int W = 1 + ((cols - 2) & ~15);
  for(int c = 1; c < W; c += 16) {
    censusOp(srow + c, cols, drow + c);
  }

  if(W != (cols - 1)) {
    censusOp(srow + cols - 1 - 16, cols, drow + cols - 1 - 16);
  }

  drow[cols-1] = 0;
}

#if defined(WITH_TBB)
struct CensusTransformFunc
{
  CensusTransformFunc(const uint8_t* src_ptr, int cols, uint8_t* dst_ptr)
      : _src(src_ptr), _cols(cols), _dst(dst_ptr) {}

  void operator()(const tbb::blocked_range<int>& range) const
  {
    for(int y = range.begin(); y != range.end(); ++y)
    {
      auto srow = _src + y*_cols;
      auto drow = _dst + y*_cols;
      census_row(srow, _cols, drow);
    }
  }

 private:
  const uint8_t* _src;
  int _cols;
  uint8_t* _dst;
}; // CensusOpRow
#endif


Image_<uint8_t> censusTransform(const Image_<uint8_t>& I, float sigma)
{
  if(sigma > 0.0) {
    // smooth the image prior to computing the census transform
  }

  Image_<uint8_t> ret(I.rows(), I.cols());

  memset(ret.data(), 0, ret.cols());
#if defined(WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(1, I.rows()-1),
                    CensusTransformFunc(I.data(), I.cols(), ret.data()));
#else
  for(int y = 1; y < I.rows() - 1; ++y) {
    auto srow = I.data() + y*I.cols;
    auto drow = ret.data() + y*I.cols;
    census_row(srow, I.cols(), drow);
  }
#endif

  memset(ret.data() + (ret.rows()-1)*ret.cols(), 0, ret.cols());

  return ret;
}

