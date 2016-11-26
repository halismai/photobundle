#ifndef PHOTOBUNDLE_TRAJECTORY_H
#define PHOTOBUNDLE_TRAJECTORY_H

#include <vector>
#include "types.h"

class Trajectory
{
 public:
  typedef int Id_t;

 public:
  Trajectory();

  //
  // append poses to the trajectory
  //
  // pose input is the estimated relative pose from VO. The class keeps track
  // of the world pose (which is the inverse of the relative pose), i.e.
  //
  // The world pose (T_w) at frame i after relative pose T_i is given by:
  //
  //  T_w_i = T_w_(i-1) * inv(T_i)
  //
  void push_back(const Mat44&, const Id_t id);

  //
  // direct element access
  //
  inline const Mat44& operator[](size_t i) const { return _data[i].pose; }
  inline       Mat44& operator[](size_t i)       { return _data[i].pose; }

  //
  // get the camera pose that is associated with the frame id. If the id is not
  // found in the trajectory, the method will assert
  //
  const Mat44& atId(const Id_t id) const;
  Mat44& atId(const Id_t id);

  //
  // get the last pose in the trajectory
  //
  inline const Mat44& back() const { return _data.back().pose; }

  // get all camera poses as 4x4 matrices
  EigenAlignedContainer_<Mat44> poses() const;

  // get all camera positions (centers) in the world
  EigenAlignedContainer_<Vec3> cameraPositions() const;

  inline size_t size() const { return _data.size(); }

 protected:
  struct PoseWithId
  {
    Mat44 pose; // 4x4 rigid body pose
    Id_t  id;    // associated frame id
  }; // PoseWithId

 private:
  EigenAlignedContainer_<PoseWithId> _data;

 private:
  void assert_unique_id(const Id_t id) const;
  typedef typename EigenAlignedContainer_<PoseWithId>::iterator Iterator_t;
  typedef typename EigenAlignedContainer_<PoseWithId>::const_iterator ConstIterator_t;

  Iterator_t find_pose_with_id(const Id_t id)
  {
    return std::find_if(std::begin(_data), std::end(_data),
                        [=](const PoseWithId& p) { return p.id == id; });
  }

  ConstIterator_t find_pose_with_id(const Id_t id) const
  {
    return std::find_if(std::begin(_data), std::end(_data),
                        [=](const PoseWithId& p) { return p.id == id; });
  }

}; // Trajectory

#endif // PHOTOBUNDLE_TRAJECTORY_H
