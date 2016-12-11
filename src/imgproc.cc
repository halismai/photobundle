#include "imgproc.h"
#include "debug.h"
#include "v128.h"

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include <stdexcept>
#include <cmath>

#if defined(WITH_OPENCV)
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <smmintrin.h>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>


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
template <typename TSrc>
struct ImageGradientFunc_
{
  ImageGradientFunc_(const TSrc* src_ptr, int cols, float* Ix_ptr, float* Iy_ptr)
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
  const TSrc* _src;
  int _cols;
  float* _Ix;
  float* _Iy;
}; // ImageGradientFunc
#endif

template <typename TSrc = uint8_t> inline
void imgradient_(const TSrc* src_ptr, const ImageSize& im_size, float* Ix_ptr, float* Iy_ptr)
{
  auto rows = im_size.rows;
  auto cols = im_size.cols;

  memset(Ix_ptr, 0.0, sizeof(float) * cols);
  memset(Iy_ptr, 0.0, sizeof(float) * cols);

#if defined(WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(1, rows-1),
                    ImageGradientFunc_<TSrc>(src_ptr, cols, Ix_ptr, Iy_ptr));
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


void imgradient(const uint8_t* src_ptr, const ImageSize& im_size, float* Ix_ptr, float* Iy_ptr)
{
  imgradient_(src_ptr, im_size, Ix_ptr, Iy_ptr);
}

void imgradient(const float* src_ptr, const ImageSize& im_size, float* Ix_ptr, float* Iy_ptr)
{
  imgradient_(src_ptr, im_size, Ix_ptr, Iy_ptr);
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


Image_<uint8_t> censusTransform(const uint8_t* I_ptr, const ImageSize& im_size)
{
  Image_<uint8_t> ret(im_size.rows, im_size.cols);
  auto dst_ptr = ret.data();

  memset(dst_ptr, 0, im_size.cols);

#if defined(WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(1, im_size.rows-1),
                    CensusTransformFunc(I_ptr, im_size.cols, dst_ptr));
#else
  for(int y = 1; y < im_size.rows - 1; ++y) {
    auto srow = I_ptr + y*im_size.cols;
    auto drow = dst_ptr + y*im_size.cols;
    census_row(srow, im_size.cols, drow);
  }
#endif

  memset(dst_ptr + (im_size.rows-1)*im_size.cols, 0, im_size.cols);
  return ret;
}

static inline
void ExtractBitPlanesChannel(const uint8_t* src, const ImageSize& image_size,
                             int bit, float* dst, float sigma)
{
  int N = image_size.rows * image_size.cols;
  for(int i = 0; i < N; ++i) {
    dst[i] = static_cast<float>( (src[i] & (1 << bit)) >> bit );
  }

  if(sigma > 0.0) {
    cv::Mat dst_(image_size.rows, image_size.cols, cv::DataType<float>::type, (void*) dst);
    cv::GaussianBlur(dst_, dst_, cv::Size(5,5), sigma, sigma);
  }
}

void computeBitPlanes(const uint8_t* image, const ImageSize& im_size,
                      EigenAlignedContainer_<Image_<float>>& dst, float sigma_ct, float sigma_bp)
{
  auto rows = im_size.rows, cols = im_size.cols;
  const uint8_t* src_ptr = nullptr;
  cv::Mat I_s;
  if(sigma_ct > 0.0f) {
    const cv::Mat I(rows, cols, cv::DataType<uint8_t>::type, (void*) image);
    cv::GaussianBlur(I, I_s, cv::Size(3, 3), sigma_ct);
    src_ptr = I_s.ptr<const uint8_t>();
  } else {
    src_ptr = image;
  }

  const auto C = censusTransform(src_ptr, im_size);

  if(dst.size() != 8)
    dst.resize(8);

#if defined(WITH_OPENMP)
#pragma omp parallel for
#endif
  for(size_t i = 0; i < 8; ++i) {
    if(dst[i].rows() != rows || dst[i].cols() != cols)
      dst[i].resize(rows, cols);
    ExtractBitPlanesChannel(C.data(), ImageSize(rows, cols), i, dst[i].data(), sigma_bp);
  }
}

void imsmooth(const float* src, const ImageSize& im_size, int ks, double s, float* dst)
{
#if defined(WITH_OPENCV)
  const cv::Mat src_(im_size.rows, im_size.cols, cv::DataType<float>::type, (void*) src);
  cv::Mat dst_(im_size.rows, im_size.cols, cv::DataType<float>::type, (void*) dst);

  cv::GaussianBlur(src_, dst_, cv::Size(ks, ks), s, s);
#else
#error "compile WITH_OPENCV"
#endif

}

void imsmooth(const uint8_t* src, const ImageSize& im_size, int ks, double s, float* dst)
{
#if defined(WITH_OPENCV)
  const cv::Mat src_(im_size.rows, im_size.cols, cv::DataType<uint8_t>::type, (void*) src);
  cv::Mat src_f;
  src_.convertTo(src_f, CV_32F);

  imsmooth(src_f.ptr<const float>(), im_size, ks, s, dst);

#else
#error "compile WITH_OPENCV"
#endif
}

template <typename T, size_t A = 16>
static bool inline is_aligned(const T* p)
{
  return 0 == (std::ptrdiff_t(p) & (A-1));
}

void disparityToDepth(const float* dmap, const ImageSize& im_size, float Bf, float* zmap)
{
  constexpr float MinValidDisparity = 0.01f;
  constexpr float InvalidDepthMark  = -.10f;

  const auto MD = _mm_set1_ps(MinValidDisparity);
  const auto IV = _mm_set1_ps(InvalidDepthMark);
  const auto BF = _mm_set1_ps(Bf);

  const auto N = im_size.numel();
  int i = 0;


  if(is_aligned(dmap) && is_aligned(zmap)) {
#if defined(WITH_OPENMP)
#pragma omp parallel for
#endif
    for(i = 0; i <= N - 4; i += 4) {
      auto d = _mm_load_ps(dmap + i);
      auto m = _mm_cmpgt_ps(d, MD);
      auto z = _mm_mul_ps(BF, _mm_rcp_ps(d));
      auto zz = _mm_blendv_ps(IV, z, m);
      _mm_store_ps(zmap + i, zz);
    }
  } else {
#if defined(WITH_OPENCV)
#pragma omp parallel for
#endif
    for(i = 0; i <= N - 4; i += 4) {
      auto d = _mm_loadu_ps(dmap + i);
      auto m = _mm_cmpgt_ps(d, MD);
      auto z = _mm_mul_ps(BF, _mm_rcp_ps(d));
      auto zz = _mm_blendv_ps(IV, z, m);
      _mm_storeu_ps(zmap + i, zz);
    }
  }

  if(N % 16) {
    for(i = N-4; i < N; ++i) {
      zmap[i] = dmap[i] > MinValidDisparity ? (Bf  * (1.0f / dmap[i])) : -1.0f;
    }
  }
}

void disparityToDepth(const cv::Mat& d, double bf, cv::Mat_<float>& z)
{
  assert( d.type() == cv::DataType<float>::type );
  z.create(d.size());

  disparityToDepth(d.ptr<const float>(), ImageSize(d.rows, d.cols), bf, z.ptr<float>());
}

