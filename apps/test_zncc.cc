#include "types.h"
#include "dataset.h"

template <class Image, class T> inline
T interp2(const Image& I, T xf, T yf, T fillval = 0.0, T offset = 0.0)
{
  const int max_cols = I.cols() - 1;
  const int max_rows = I.rows() - 1;

  xf += offset;
  yf += offset;

  int xi = (int) std::floor(xf);
  int yi = (int) std::floor(yf);

  xf -= xi;
  yf -= yi;

  if( xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows )
  {
    const T wx = 1.0 - xf;
    return (1.0 - yf) * ( I(yi,   xi)*wx + I(yi,   xi+1)*xf )
               +  yf  * ( I(yi+1, xi)*wx + I(yi+1, xi+1)*xf );
  } else
  {
    if( xi == max_cols && yi < max_rows )
      return ( xf > 0 ) ? fillval : (1.0-yf)*I(yi,xi) + yf*I(yi+1, xi);
    else if( yi == max_rows && xi < max_cols )
      return ( yf > 0 ) ? fillval : (1.0-xf)*I(yi,xi) + xf*I(yi, xi+1);
    else if( xi == max_cols && yi == max_rows )
      return ( xf > 0 || yf > 0 ) ? fillval : I(yi, xi);
    else
      return fillval;
  }
}


template <int N> constexpr int square() { return N*N; }

template <int R, class ImageType, class ProjType, typename T = double>
void interpolateFixedPatch(Vec_<T, square<2*R+1>()>& dst,
                           const ImageType& I, const ProjType& p,
                           const T& fillval = T(0), const T& offset = T(0))
{
  const T x = static_cast<T>( p[0] + offset );
  const T y = static_cast<T>( p[1] + offset );

  auto d_ptr = dst.data();
  for(int r = -R; r <= R; ++r) {
    for(int c = -R; c <= R; ++c) {
      *d_ptr++ = interp2(I, c + x, r + y, fillval);
    }
  }
}

template <int R, typename T = float>
class ZnccPatch_
{
  static_assert(std::is_floating_point<T>::value, "T must be floating point");

 public:
  static constexpr int Radius = R;
  static constexpr int Dimension = (2*R+1) * (2*R+1);

 public:
  inline ZnccPatch_() {}

  template <class ImageType, class ProjType> inline
  ZnccPatch_(const ImageType& image, const ProjType& uv) { set(image, uv); }

  template <class ImageType, class ProjType> inline
  const ZnccPatch_& set(const ImageType& I, const ProjType& uv)
  {
    interpolateFixedPatch<R>(_data, I, uv, T(0.0), T(0.0));
    T mean = _data.array().sum() / (T) _data.size();
    _data.array() -= mean;
    _norm = _data.norm();

    return *this;
  }

  template <class ImageType, class ProjType>
  inline static ZnccPatch_ FromImage(const ImageType& I, const ProjType& p)
  {
    ZnccPatch_ ret;
    ret.set(I, p);
    return ret;
  }

  inline T score(const ZnccPatch_& other) const
  {
    T d = _norm * other._norm;
    return d > 1e-6 ? _data.dot(other._data) / d : -1.0;
  }

 private:
  Vec_<T, Dimension> _data;
  T _norm;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // ZnccPatch

int main()
{
  auto dataset = Dataset::Create("../config/kitti_stereo.cfg");

  auto frame = dataset->getFrame(0);
  auto rows = frame->image().rows, cols = frame->image().cols;
  Eigen::Map<const Image_<uint8_t>> I(frame->image().data, rows, cols);

  for(int y = 5; y < rows - 6; ++y) {
    for(int x = 5; x < cols - 6; ++x) {
      Vec_<int,2> uv(x, y);
      ZnccPatch_<2, float> patch(I, uv);
      auto score = patch.score(patch);
      auto err = std::abs(1.0 - score);
      if(err > 1e-6) {
        if(score != -1.0)
          printf("error %g\n", err);
      }
    }
  }
}

