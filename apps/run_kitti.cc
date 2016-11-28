#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "photobundle.h"
#include "utils.h"
#include "debug.h"
#include "dataset.h"
#include "imgproc.h"
#include "pose_utils.h"


void disparityToDepth(const cv::Mat& d, double bf, cv::Mat_<float>& z)
{
  assert( d.type() == cv::DataType<float>::type );
  z.create(d.size());

  disparityToDepth(d.ptr<const float>(), ImageSize(d.rows, d.cols), bf, z.ptr<float>());
}


int main(int argc, char** argv)
{
  utils::ProgramOptions options;
  options
      ("config,c", "../config/kitti_stereo.cfg", "config file")
      .parse(argc, argv);

  utils::ConfigFile cf(options.get<std::string>("config"));
  auto dataset = Dataset::Create(options.get<std::string>("config"));
  auto Bf = dataset->calibration().b() * dataset->calibration().fx();
  auto T_init = loadPosesKittiFormat(cf.get<std::string>("trajectory"));

  PhotometricBundleAdjustment photoba(dataset->calibration(), dataset->imageSize(), {cf});

  cv::Mat_<float> zmap;
  UniquePointer<DatasetFrame> frame;
  for(int f_i = 0; (frame = dataset->getFrame(f_i)); ++f_i) {
    disparityToDepth(frame->disparity(), Bf, zmap);
    // TODO add frame to bundle adjustment
  }


  writePosesKittiFormat("poses.txt", T_init); // TODO
  return 0;
}

