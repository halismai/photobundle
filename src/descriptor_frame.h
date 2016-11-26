#ifndef PHOTOBUNDLE_DESCRIPTOR_FRAME_H
#define PHOTOBUNDLE_DESCRIPTOR_FRAME_H

#include "types.h"
#include "imgproc.h"

#include <cassert>

template <class ImageT>
class ImageGradient_
{
 public:
  typedef ImageT Image;

 public:
  ImageGradient_() = default;
  ImageGradient_(const ImageT& Ix, const ImageT& Iy)
      : _Ix(Ix), _Iy(Iy) {}

  inline const ImageT& Ix() const { return _Ix; }
  inline const ImageT& Iy() const { return _Iy; }

  inline ImageT absGradientMag() const
  {
    return _Ix.array().abs() + _Iy.array().abs();
  }

 private:
  ImageT _Ix;
  ImageT _Iy;
}; // ImageGradient

template <class ImageT>
class DescriptorFrame_
{
 public:
  typedef ImageT Image;
  typedef EigenAlignedContainer_<ImageT> ImageList;
  typedef ImageGradient_<ImageT> ImageGradientType;
  typedef EigenAlignedContainer_<ImageGradientType> ImageGradientList;

 public:
  enum class Type
  {
    Intensity,
    IntensityAndGradient,
    BitPlanes
  }; // Type

 public:
  /**
   * \param frame_id the frame number (unique per image)
   * \param I        grayscale input image
   * \param channels list of descriptor channels
   * \param gx       list of x-gradients per channel
   * \param gy       list of y-gradients per channel
   */
  DescriptorFrame_(uint32_t frame_id, const ImageT& I, const ImageList& channels,
                  const ImageList& gx, const ImageList& gy)
      : _frame_id(frame_id),
      _max_rows( channels.front().rows() - 1 ),
      _max_cols( channels.front().cols() - 1 ),
      _image( I ),
      _channels( channels )
  {
    assert( channels.size() == gx.size() && gx.size() == gy.size() && "size mismatch");

    _gradients.resize(_channels.size());
    for(size_t i = 0; i < _channels.size(); ++i) {
      _gradients[i] = ImageGradientType(gx[i], gy[i]);
    }
  }


  DescriptorFrame_(const DescriptorFrame_&) = delete;
  DescriptorFrame_& operator=(const DescriptorFrame_&) = delete;

  inline uint32_t id() const { return _frame_id; }

  inline bool operator<(const DescriptorFrame_& other) const
  {
    return _frame_id < other._frame_id;
  }

  inline size_t numChannels() const { return _channels.size(); }

  inline const Image& getChannel(size_t i) const
  {
    assert(i < _channels.size());
    return _channels[i];
  }

  inline const ImageGradientType& getChannelGradient(size_t i) const
  {
    assert(i < _gradients.size());
    return _gradients[i];
  }

  inline const Image& getOriginalImage() const { return _image; }

  /**
   * \return true of point projects to the image
   */
  template <class ProjType> inline
  bool isProjectionValid(const ProjType& x) const
  {
    return x[0] >= 0.0 && x[0] < _max_cols &&
           x[1] >= 0.0 && x[1] < _max_rows;
  }

  void computeSaliencyMap(typename ImageGradientType::Image& smap) const
  {
    smap.array() = _gradients[0].absGradientMag();

    for(size_t i = 1; i < _gradients.size(); ++i) {
      smap.array() += _gradients[i].absGradientMag();
    }
  }

 public:
  /**
   * Interface to create different descriptor types
   */
  static UniquePointer<DescriptorFrame_<ImageT>> Create(uint32_t id, const ImageT& I, Type type)
  {
    switch(type)
    {
      case Type::Intensity:
        {
          return make_unique<DescriptorFrame_<ImageT>>(
              id, I, {I}, {xgradient<float>(I)}, {ygradient<float>{I}});
        } break;

        case Type::IntensityAndGradient
        {
          ImageList channels{I, xgradient<float>(I), ygradient<float>(I)};
          ImageList Gx, Gy;
          computeGradientsMultiChannel(channels, Gx, Gy);

          return make_unique<DescriptorFrame_<ImageT>>(
              id, I, channels, Gx, Gy);
        } break;

      case Type::BitPlanes:
        {
          auto channels = computeBitPlanes(I);
        } break;
    }
  }

 private:
  uint32_t _frame_id;
  uint32_t _max_rows;
  uint32_t _max_cols;

  Image _image;
  ImageList _channels;
  ImageGradientList _gradients;
}; // DescriptorFrame

#endif // PHOTOBUNDLE_DESCRIPTOR_FRAME_H
