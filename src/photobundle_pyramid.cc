#include "photobundle_pyramid.h"
#include "debug.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/LU>

PhotometricBundleAdjustmentPyr::
PhotometricBundleAdjustmentPyr(int num_levels, const Calibration& calib,
                               const ImageSize& im_size, const Options& options)
  : _rows(im_size.rows), _cols(im_size.cols)
{
  assert( num_levels > 0 );


  Calibration calib_pyr(calib);
  ImageSize im_size_pyr(im_size);

  _pyr.resize(num_levels);
  for(size_t i = 1; i < _pyr.size(); ++i) {
    _pyr[i].reset(new PhotometricBundleAdjustment(calib_pyr, im_size_pyr, options));
    calib_pyr = calib_pyr.pyrDown();
    im_size_pyr = im_size_pyr.pyrDown();
  }

  _im_pyr.resize(num_levels);
  for(size_t i = 0; i < _im_pyr.size(); ++i) {
    _im_pyr[i].reset(new cv::Mat);
  }

  _z_pyr.resize(num_levels);
  for(size_t i = 0; i < _im_pyr.size(); ++i) {
    _z_pyr[i].reset(new cv::Mat);
  }
}

void PhotometricBundleAdjustmentPyr::
addFrame(const uint8_t* image, const float* depth, const Mat44& T, Result* result)
{
  Warn("This is not finished yet\n");

  {
    const cv::Mat I(_rows, _cols, CV_8U, (void*) image);
    *_im_pyr[0] = I;
    for(size_t i = 1; i < _im_pyr.size(); ++i) {
      cv::pyrDown(*_im_pyr[i-1], *_im_pyr[i]);
    }
  }

  {
    // for depth, we'll do nearest neighbor resize
    const cv::Mat Z(_rows, _cols, CV_32F, (void*) depth);
    *_z_pyr[0] = Z;
    for(size_t i = 1; i < _z_pyr.size(); ++i) {
      cv::resize(*_z_pyr[i-1], *_z_pyr[i], _im_pyr[i]->size());
    }
  }

  Result tmp;
  Mat44 T_init(T);
  for(int i = (int) _im_pyr.size(); i >= 0; --i) {
    Info("Pyramid level %d\n", i);
    _pyr[i]->addFrame(_im_pyr[i]->ptr<uint8_t>(), _z_pyr[i]->ptr<float>(), T_init, &tmp);
    T_init = tmp.poses.back().inverse();
  }

  if(result)
    *result = tmp;
}

