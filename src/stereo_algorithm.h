#ifndef PHOTOBUNDLE_STEREO_ALG_H
#define PHOTOBUNDLE_STEREO_ALG_H

#include "types.h"

namespace cv {
class Mat;
}; // cv

namespace utils {
class ConfigFile;
}; // utils

class StereoAlgorithm
{
 public:
  StereoAlgorithm(const utils::ConfigFile&);
  StereoAlgorithm(std::string conf_fn);

  virtual ~StereoAlgorithm();

  /**
   * \param left the left image
   * \param right the right image
   * \param dmap disparity map
   */
  void run(const cv::Mat& left, const cv::Mat& right, cv::Mat& dmap);

  /**
   * \return the invalid value as a floating point number
   */
  float getInvalidValue() const;

 protected:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // StereoAlgorithm


#endif // PHOTOBUNDLE_STEREO_ALG_H
