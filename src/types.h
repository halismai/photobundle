#ifndef PHOTOBUNDLE2_TYPES_H
#define PHOTOBUNDLE2_TYPES_H

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <iosfwd>

template <typename T, int M = Eigen::Dynamic, int N = Eigen::Dynamic>
using Mat_ = Eigen::Matrix<T,M,N>;

template <typename T, int M = Eigen::Dynamic>
using Vec_ = Eigen::Matrix<T,M,1>;

template <typename T, int N = Eigen::Dynamic>
using RowVec_ = Eigen::Matrix<T,1,N>;

template <typename T>
using Image_ = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

typedef Mat_<double,3,3> Mat33;
typedef Mat_<double,4,4> Mat44;
typedef Mat_<double,3,4> Mat34;
typedef Mat_<double,3,Eigen::Dynamic> Mat3X;
typedef Mat_<double,4,Eigen::Dynamic> Mat4X;

typedef Vec_<double,2> Vec2;
typedef Vec_<double,3> Vec3;
typedef Vec_<double,4> Vec4;

template <class EigenType>
using EigenAllocator_ = Eigen::aligned_allocator<EigenType>;

template <class EigenType, template <class, class> class Container = std::vector>
using EigenAlignedContainer_ = std::vector<EigenType, EigenAllocator_<EigenType>>;

template <typename _T> using
SharedPointer = std::shared_ptr<_T>;

template <typename _T> using
UniquePointer = std::unique_ptr<_T>;

template <class _T, class ... Args> inline
UniquePointer<_T> make_unique(Args&& ... args) {
#if __cplusplus > 201103L
  return std::make_unique<_T>(std::forward<Args>(args)...);
#else
  return UniquePointer<_T>(new _T(std::forward<Args>(args)...));
#endif
}

template <class _T, class ... Args> inline
SharedPointer<_T> make_shared(Args&& ... args) {
  return std::make_shared<_T>(std::forward<Args>(args)...);
}

struct ImageSize
{
  int rows = 0;
  int cols = 0;

  inline ImageSize(int r = 0, int c = 0)
  : rows(r), cols(c) {}

  inline int numel() const { return rows*cols; }
  inline int area() const { return numel(); }

  inline bool empty() const { return 0 == numel(); }

  inline ImageSize pyrDown() const
  {
    return ImageSize((rows + 1) / 2, (cols + 1) / 2);
  }

 private:
  friend std::ostream& operator<<(std::ostream&, const ImageSize&);
}; // ImageSize

#endif
