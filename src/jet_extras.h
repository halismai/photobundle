// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//

// modified slightly hy halismai to eliminate some warnings
#ifndef CERES_PUBLIC_JET_EXTRAS_H_
#define CERES_PUBLIC_JET_EXTRAS_H_

#include "ceres/jet.h"
#include "Eigen/Core"

namespace ceres {

// A jet traits class to make it easier to work with mixed auto / numeric diff.
template<typename T>
struct JetOps {
  static bool IsScalar() {
    return true;
  }
  static T GetScalar(const T& t) {
    return t;
  }
  static void SetScalar(const T& scalar, T* t) {
    *t = scalar;
  }
  static void ScaleDerivative(double /*scale_by*/, T * /*value*/) {
    // For double, there is no derivative to scale.
  }
};

template<typename T, int N>
struct JetOps<Jet<T, N> > {
  static bool IsScalar() {
    return false;
  }
  static T GetScalar(const Jet<T, N>& t) {
    return t.a;
  }
  static void SetScalar(const T& scalar, Jet<T, N>* t) {
    t->a = scalar;
  }
  static void ScaleDerivative(double scale_by, Jet<T, N> *value) {
    value->v *= scale_by;
  }
};

template<typename FunctionType, int kNumArgs, typename ArgumentType>
struct Chain {
  static ArgumentType Rule(const FunctionType &f,
                           const FunctionType /*dfdx*/[kNumArgs],
                           const ArgumentType /*x*/[kNumArgs]) {
    // In the default case of scalars, there's nothing to do since there are no
    // derivatives to propagate.
    return f;
  }
};

// XXX Add documentation here!
template<typename FunctionType, int kNumArgs, typename T, int N>
struct Chain<FunctionType, kNumArgs, Jet<T, N> > {
  static Jet<T, N> Rule(const FunctionType &f,
                        const FunctionType dfdx[kNumArgs],
                        const Jet<T, N> x[kNumArgs]) {
    // x is itself a function of another variable ("z"); what this function
    // needs to return is "f", but with the derivative with respect to z
    // attached to the jet. So combine the derivative part of x's jets to form
    // a Jacobian matrix between x and z (i.e. dx/dz).
    Eigen::Matrix<T, kNumArgs, N> dxdz;
    for (int i = 0; i < kNumArgs; ++i) {
      dxdz.row(i) = x[i].v.transpose();
    }

    // Map the input gradient dfdx into an Eigen row vector.
    Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);

    // Now apply the chain rule to obtain df/dz. Combine the derivative with
    // the scalar part to obtain f with full derivative information.
    Jet<T, N> jet_f;
    jet_f.a = f;
    jet_f.v = vector_dfdx.template cast<T>() * dxdz;  // Also known as dfdz.
    return jet_f;
  }
};

}  // namespace ceres

#endif  // CERES_PUBLIC_JET_EXTRAS_H_

