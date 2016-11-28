#ifndef POSE_UTILS_H
#define POSE_UTILS_H

#include "types.h"

typedef EigenAlignedContainer_<Mat44> PoseList;

/**
 * \return list of poses from a file
 */
PoseList loadPosesKittiFormat(std::string);

/**
 * Writes a list of poses to a file using the KITTI benchmark format
 * \return true on success
 */
bool writePosesKittiFormat(std::string, const PoseList&);

/**
 * \param input poses in 'global' coordinate system. i.e. if you plot the camera
 * centers (the translfation part of each pose) you will get the camera
 * trajectory
 *
 * \return Relative poses (local) between every camera
 */
PoseList convertPoseToLocal(const PoseList&);

/**
 * Reports performance using the KITTI benchmark meteric
 */
void RunKittiEvaluation(std::string ground_truth_dir, std::string results_dir,
                        std::string output_prefix);

#endif // POSE_UTILS_H
