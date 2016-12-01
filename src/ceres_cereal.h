#ifndef CERES_CEREAL_H
#define CERES_CEREAL_H

#if defined(WITH_CEREAL)
#include <ceres/iteration_callback.h>

namespace cereal {

template <class Archive> inline
void save(Archive& ar, const ceres::IterationSummary& s)
{
  ar(
      s.iteration,
      s.step_is_valid,
      s.step_is_nonmonotonic,
      s.step_is_successful,
      s.cost,
      s.cost_change,
      s.gradient_max_norm,
      s.gradient_norm,
      s.step_norm,
      s.eta,
      s.step_size,
      s.line_search_function_evaluations,
      s.line_search_gradient_evaluations,
      s.line_search_iterations,
      s.linear_solver_iterations,
      s.iteration_time_in_seconds,
      s.step_solver_time_in_seconds,
      s.cumulative_time_in_seconds);
}

// we do not need separate functions for this becaus of the POD, RTFM
template <class Archive> inline
void load(Archive& ar, ceres::IterationSummary& s)
{
  ar(
      s.iteration,
      s.step_is_valid,
      s.step_is_nonmonotonic,
      s.step_is_successful,
      s.cost,
      s.cost_change,
      s.gradient_max_norm,
      s.gradient_norm,
      s.step_norm,
      s.eta,
      s.step_size,
      s.line_search_function_evaluations,
      s.line_search_gradient_evaluations,
      s.line_search_iterations,
      s.linear_solver_iterations,
      s.iteration_time_in_seconds,
      s.step_solver_time_in_seconds,
      s.cumulative_time_in_seconds);
}

}; // cereal

#endif
#endif // CERES_CEREAL_H
