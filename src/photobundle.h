#ifndef PHOTOBUNDLE_PHOTOBUNDLE_H
#define PHOTOBUNDLE_PHOTOBUNDLE_H

#include <ceres/iteration_callback.h>

#include "types.h"
#include "trajectory.h"
#include "calibration.h"

#include <iosfwd>

#include <boost/circular_buffer.hpp>

struct ImageSize
{
  int rows = 0;
  int cols = 0;

  inline ImageSize() {}
}; // ImageSize

class PhotometricBundleAdjustment
{
 public:
  /**
   */
  struct Options
  {
    /** number of frames in the sliding window */
    int slidingWindowSize = 5;

    /** radius of the image patch */
    int patchRadius = 2;

    /** radius (side length) of an area to prevent intializing new points when a
     * new frame is added */
    int maskBlockRadius = 1;

    /** Maximum distance (age) to keep a scene point in the system */
    int maxFrameDistance = 1;

    /** number of threads to use in the solve (-1 means maximum threads) */
    int numThreads = -1;

    /** optional gaussian weighting to focus on the center of the patch */
    bool doGaussianWeighting = false;

    /** minimum score to verify if a scene point exists in a new frame. This is
     * the ZNCC score which is [-1, 1] */
    double minScore = 0.75;

    Options() {}

   private:
    friend std::ostream& operator<<(std::ostream&, const Options&);
  }; //  Options


  struct Result
  {
    /** refined world poses */
    EigenAlignedContainer_<Mat44> poses;

    /** refined world points */
    EigenAlignedContainer_<Vec3> refinedPoints;

    /** the original points for comparison */
    EigenAlignedContainer_<Vec3> originalPoints;

    //
    // optimization statistics
    //
    double initialCost  = -1.0; //< objective at the first start
    double finalcost    = -1.0; //< objective at termination
    double fixedCost    = -1.0; //< fixed cost not included in optimization

    int numSuccessfulStep = 0; //< number of successfull optimizer steps
    int numResiduals      = 0; //< number of residuals in the problem

    std::string message; //< optimizer message

    /**
     * Iteration details from ceres
     * iterationSummary.size() is the total number of iterations
     */
    std::vector<ceres::IterationSummary> iterationSummary;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  }; // Result


 public:
  /**
   * \param the camera calibration (pinhole model)
   * \param the image size
   * \param options for the algorithm
   */
  PhotometricBundleAdjustment(const Calibration&, const ImageSize&, const Options& = Options());

  /**
   */
  ~PhotometricBundleAdjustment();

  /**
   * \param image pointer to the image
   * \param depth_map pointer to the depth map
   * \param T pose initialization for this frame
   * \param result, if not null we store the optmization results in it
   */
  void addFrame(const uint8_t* image, const float* depth_map, const Mat44& T, Result* = nullptr);

 private:
  struct ScenePoint;
  typedef UniquePointer<ScenePoint>      ScenePointPointer;
  typedef std::vector<ScenePointPointer> ScenePointPointerList;

  /** removes scene points whose frame id == id */
  ScenePointPointerList removePointsAtFrame(uint32_t id);

  struct DescriptorFrame;
  typedef UniquePointer<DescriptorFrame>                 DescriptorFramePointer;
  typedef boost::circular_buffer<DescriptorFramePointer> DescriptorFrameBuffer;

  /** \return the frame data at the given id */
  const DescriptorFrame* getFrameAtId(uint32_t id) const;

 private:
  uint32_t _frame_id = 0;

  Calibration _calib;
  ImageSize   _image_size;
  Options     _options;
  Trajectory _trajectory;
  DescriptorFrameBuffer _frame_buffer;
  ScenePointPointerList _scene_points;
}; // PhotometricBundleAdjustment

#endif // PHOTOBUNDLE_PHOTOBUNDLE_H
