#ifndef IMGPROC_H
#define IMGPROC_H

#include "types.h"
#include <type_traits>

namespace cv {
class Mat;
template <typename T> class Mat_;
}; //cv

/**
 * central difference image gradient
 */
void imgradient(const uint8_t* src, const ImageSize&, float* Ix, float* Iy);
void imgradient(const float* src, const ImageSize&, float* Ix, float *Iy);


/**
 * Gaussian smoothing
 */
void imsmooth(const uint8_t* src, const ImageSize&, int kernel_size, double sigma, float* dst);
void imsmooth(const float* src, const ImageSize&, int kernel_size, double sigma, float* dst);

/**
 * converts a disparity image to depth
 *
 * Invalid disparities are <= 0
 * Invalid depth will be <= 0 as well
 *
 * \param dmap pointer to the disparity map
 * \param im_size the image size
 * \param Bf the steroe baseline * the focal length (in pixels)
 * \param depth output depth
 */
void disparityToDepth(const float* dmap, const ImageSize& im_size, float Bf, float* depth);

void disparityToDepth(const cv::Mat& disparity, double Bf, cv::Mat_<float>& depth);


/**
 * bitplanes
 */
void computeBitPlanes(const uint8_t* I_ptr, const ImageSize& im_size,
                      EigenAlignedContainer_<Image_<float>>& dst,
                      float sigma_ct = 1.0, float sigma_bp = 1.5);

template <typename T> static inline constexpr
T imgradient_scale(typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
  return T(1);
}

template <typename T> static inline constexpr
T imgradient_scale(typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
{
  return T(0.5);
}

template <typename TDst, typename TSrc> inline
void xgradient(const TSrc* src, int rows, int cols, TDst* dst)
{
  static_assert(std::is_signed<TDst>::value, "TDst must be signed");
  constexpr auto S = imgradient_scale<TDst>();

  using namespace Eigen;
  typedef Mat_<TSrc, Dynamic, Dynamic> SrcMat;
  typedef Mat_<TDst, Dynamic, Dynamic> DstMat;

  typedef Map<const SrcMat> SrcMap;
  typedef Map<DstMat>       DstMap;

  DstMap Ix(dst, rows, cols);
  const SrcMap I(src, rows, cols);

  Ix.col(0) = S * (I.col(1).template cast<TDst>() - I.col(0).template cast<TDst>());

  Ix.block(0, 1, rows, cols - 2) =
      S * (I.block(0, 2, rows, cols - 2).template cast<TDst>() -
           I.block(0, 0, rows, cols - 2).template cast<TDst>());

  Ix.col(cols-1) = S * (I.col(cols-1).template cast<TDst>() -
                        I.col(cols-2).template cast<TDst>());
}

template <typename TDst, typename TSrc> inline
Mat_<TDst,Eigen::Dynamic,Eigen::Dynamic>
xgradient(const TSrc* src, int rows, int cols)
{
  Mat_<TDst, Eigen::Dynamic, Eigen::Dynamic> dst(rows, cols);
  xgradient(src, rows, cols, dst.data());

  return dst;
}

template <typename TDst, class Derived> inline
Mat_<TDst, Eigen::Dynamic, Eigen::Dynamic>
xgradient(const Eigen::DenseBase<Derived>& M)
{
  return xgradient<TDst>(&M(0,0), M.rows(), M.cols());
}

template <typename TDst, typename TSrc> inline
void ygradient(const TSrc* src, int rows, int cols, TDst* dst)
{
  static_assert(std::is_signed<TDst>::value, "TDst must be signed");
  constexpr auto S = imgradient_scale<TDst>();

  using namespace Eigen;
  typedef Mat_<TSrc, Dynamic, Dynamic> SrcMat;
  typedef Mat_<TDst, Dynamic, Dynamic> DstMat;

  typedef Map<const SrcMat> SrcMap;
  typedef Map<DstMat>       DstMap;

  DstMap Iy(dst, rows, cols);
  const SrcMap I(src, rows, cols);

  Iy.row(0) = S * (I.row(1).template cast<TDst>() - I.row(0).template cast<TDst>());

  Iy.block(1, 0, rows - 2, cols) =
      S * (I.block(2, 0, rows - 2, cols).template cast<TDst>() -
           I.block(0, 0, rows - 2, cols).template cast<TDst>());

  Iy.row(rows - 1) = S * (I.row(rows - 1).template cast<TDst>() -
                          I.row(rows - 2).template cast<TDst>());
}

template <typename TDst, typename TSrc> inline
Mat_<TDst,Eigen::Dynamic,Eigen::Dynamic>
ygradient(const TSrc* src, int rows, int cols)
{
  Mat_<TDst, Eigen::Dynamic, Eigen::Dynamic> dst(rows, cols);
  ygradient(src, rows, cols, dst.data());

  return dst;
}

template <typename TDst, class Derived> inline
Mat_<TDst, Eigen::Dynamic, Eigen::Dynamic>
ygradient(const Eigen::DenseBase<Derived>& M)
{
  return ygradient<TDst>(&M(0,0), M.rows(), M.cols());
}

template <typename TDst, typename TSrc> inline
void imgradientAbsMag(const TSrc* src, int rows, int cols, TDst* dst)
{
  using namespace Eigen;

  typedef Map< Mat_<TDst,Dynamic,Dynamic> > DstMap;
  DstMap G(dst, rows, cols);

  G = (xgradient<TDst>(src, rows, cols).array().abs() +
       ygradient<TDst>(src, rows, cols).array().abs());
}

template <typename TDst, typename TSrc> inline
Mat_<TDst,Eigen::Dynamic,Eigen::Dynamic>
imgradientAbsMag(const TSrc* src, int rows, int cols)
{
  Mat_<TDst,Eigen::Dynamic,Eigen::Dynamic> ret(rows, cols);
  imgradientAbsMag<TDst>(src, rows, cols, ret.data());
  return ret;
}

template <typename TDst, class Derived> inline
Mat_<TDst,Eigen::Dynamic,Eigen::Dynamic>
imgradientAbsMag(const Eigen::DenseBase<Derived>& A)
{
  return imgradientAbsMag<TDst>(&A(0,0), A.rows(), A.cols());
}


template <class Image, class Mask>
class IsLocalMax_
{
  typedef typename Image::Scalar T;

 public:
  IsLocalMax_(const Image& image, const Mask& mask, int radius, T min_saliency = T(0))
      : _I(image), _mask(mask), _radius(radius), _min_saliency(min_saliency)
  {
    assert( _I.rows() == _mask.rows() && _I.cols() == _mask.cols() &&
           "image and mask must have the same size");
  }

  inline bool operator()(int row, int col) const
  {
    if(_radius > 0) {
      auto v = _I(row, col);
      if(!_mask(row, col) || v < _min_saliency)
        return false;

      for(int r = -_radius; r <= _radius; ++r) {
        for(int c = -_radius; c <= _radius; ++c) {
          if(!(!r && !c) && _I(r+row,c+col) >= v) {
            return false;
          }
        }
      }
    }

    return true;
  }

 private:
  const Image& _I;
  const Mask& _mask;
  int _radius;
  T _min_saliency;
}; // IsLocalMax


template <class ImageList> inline
void computeGradientsMultiChannel(const ImageList& channels, ImageList& Gx, ImageList& Gy)
{
  auto N = channels.size();
  Gx.resize(N);
  Gy.resize(N);
  for(size_t i = 0; i < N; ++i) {
    Gx[i] = xgradient<float>(channels[i]);
    Gy[i] = ygradient<float>(channels[i]);
  }
}

#endif // IMGPROC_H

