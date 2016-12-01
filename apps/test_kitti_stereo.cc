#include "dataset.h"
#include "stereo_algorithm.h"
#include "calibration.h"
#include "utils.h"
#include "imgproc.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <fstream>
#include <Eigen/LU>

void colorizeDisparity(const cv::Mat& src, cv::Mat& dst, double min_d, double num_d)
{
  THROW_ERROR_IF( src.type() != cv::DataType<float>::type, "disparity must be float" );

  double scale = 0.0;
  if(num_d > 0) {
    scale = 255.0 / num_d;
  } else {
    double max_val = 0;
    cv::minMaxLoc(src, nullptr, &max_val);
    scale = 255.0 / max_val;
  }

  src.convertTo(dst, CV_8U, scale);
  cv::applyColorMap(dst, dst, cv::COLORMAP_JET);

  for(int y = 0; y < src.rows; ++y)
    for(int x = 0; x < src.cols; ++x)
      if(src.at<float>(y,x) <= min_d)
        dst.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
}

inline cv::Mat colorizeDisparity(const cv::Mat& src, double min_d = 0, double num_d = -1)
{
  cv::Mat ret;
  colorizeDisparity(src, ret, min_d, num_d);
  return ret;
}


void overlayDisparity(const cv::Mat& I, const cv::Mat& D, cv::Mat& dst,
                      double alpha, double min_d = 0, double num_d = 128)
{
  cv::Mat image;
  switch( I.type() )
  {
    case CV_8UC1:
      cv::cvtColor(I, image, CV_GRAY2BGR);
      break;
    case CV_8UC3:
      image = I;
      break;
    case CV_8UC4:
      cv::cvtColor(I, image, CV_BGRA2BGR);
      break;
    default:
      THROW_ERROR("unsupported image type");
  }

  cv::addWeighted(image, alpha, colorizeDisparity(D, min_d, num_d), 1.0-alpha, 0.0, dst);
}


int main(int argc, char** argv)
{
  utils::ProgramOptions options;
  options("config,c", "../config/kitti_stereo.cfg", "config file").parse(argc, argv);

  auto dataset = Dataset::Create(options.get<std::string>("config"));

  cv::Mat dmap;
  UniquePointer<DatasetFrame> frame;
  int f_i = 0;
  while( (frame = dataset->getFrame(f_i++)) ) {
    fprintf(stdout, "Frame %06d\r", f_i-1); fflush(stdout);
    overlayDisparity(frame->image(), frame->disparity(), dmap, 0.7);
    cv::imshow("dmap", dmap);
    if('q' == (0xff & cv::waitKey(5)))
      break;
  }

  auto calib = dataset->calibration();

  std::vector<float> z( dataset->imageSize().numel() );

  disparityToDepth(frame->disparity().ptr<float>(), dataset->imageSize(),
                   calib.baseline() * calib.fx(), z.data());

  Mat33 K_inv = calib.K().inverse();
  std::ofstream ofs("X");
  for(int y = 0, i = 0; y < frame->disparity().rows; ++y) {
    for(int x = 0; x < frame->disparity().cols; ++x, ++i) {
      if(z[i] > 0.0 && z[i] < 10.0) {
        Vec_<double,3> X = z[i] * K_inv * Vec_<double,3>(x, y, 1.0);
        ofs << X.transpose() << "\n";
      }
    }
  }

  return 0;
}


