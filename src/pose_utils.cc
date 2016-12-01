#include "pose_utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <Eigen/Geometry>

PoseList loadPosesKittiFormat(std::string fn)
{
  std::ifstream ifs(fn);
  if(!ifs.is_open()) {
    throw std::runtime_error("failed to open pose file");
  }

  PoseList ret;
  while(!ifs.eof()) {
    std::string line;
    std::getline(ifs, line);

    std::stringstream ss(line);
    double vals[12];
    if(!line.empty()) {
      for(int i = 0; i < 12; ++i) {
        ss >> vals[i];
      }

      Mat44 T(Mat44::Identity());
      for(int i = 0, c=0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
          T(i,j) = vals[c++];
        }
      }

      ret.push_back( T );
    }
  }

  return ret;
}


bool writePosesKittiFormat(std::string fn, const PoseList& T)
{
  std::ofstream ofs(fn);
  if(!ofs.is_open())
    return false;

  for(size_t i = 0; i < T.size(); ++i) {
    for(int r = 0; r < 3; ++r) {
      for(int c = 0; c < 4; ++c) {
        ofs << (T[i](r,c)) << " ";
      }
    }
    ofs << "\n";
  }

  return true;
}


PoseList convertPoseToLocal(const PoseList& T_w)
{
  if(T_w.empty())
    throw std::runtime_error("no poses");

  PoseList T_i(T_w.size());
  T_i[0] = Eigen::Isometry3d( T_w[0] ).inverse().matrix();
  for(size_t i = 1; i < T_w.size(); ++i) {
    T_i[i] = Eigen::Isometry3d(T_w[i]).inverse().matrix() * T_w[i-1];
  }

  return T_i;
}
