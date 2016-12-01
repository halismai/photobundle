#ifndef PHOTOBUNDLE_PHOTOBUNDLE_PYR_H
#define PHOTOBUNDLE_PHOTOBUNDLE_PYR_H

#include "photobundle.h"

namespace cv {
class Mat;
}; // cv

/**
 * Pyramid implementation
 */
class PhotometricBundleAdjustmentPyr
{
 public:
  typedef PhotometricBundleAdjustment::Options Options;
  typedef PhotometricBundleAdjustment::Result Result;

 public:
  /**
   * \param num_levels number of levels in the pyramid
   * \param calibation camera calibration at the finest level
   * \param imageSize  image size at the finest level
   * \param options    algorithm options
   */
  PhotometricBundleAdjustmentPyr(int num_levels, const Calibration& calib,
                                 const ImageSize&, const Options& = Options());

  ~PhotometricBundleAdjustmentPyr();

  /**
   * \param image  pointer to the image at the finest frame
   * \param depth  pointer to the depth map at finest frame
   * \param T      pose initialization
   * \param result the optimization result
   */
  void addFrame(const uint8_t* image, const float* depth_map, const Mat44& T, Result* = nullptr);

 private:
  int _rows, _cols; // image size at the finest level

  typedef UniquePointer<PhotometricBundleAdjustment> PhotometricBundleAdjustmentPoitner;
  std::vector<PhotometricBundleAdjustmentPoitner> _pyr;

  typedef UniquePointer<cv::Mat> CvMatPointer;
  std::vector<CvMatPointer> _im_pyr;
  std::vector<CvMatPointer> _z_pyr;
}; // PhotometricBundleAdjustmentPyr

#endif // PHOTOBUNDLE_PHOTOBUNDLE_PYR_H
