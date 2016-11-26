// Copyright (c) 2012 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// Author: mierle@google.com (Keir Mierle)
//

// Modified by halismai@cs.cmu.edu to work with Eigen


#ifndef SAMPLE_EIGEN_H
#define SAMPLE_EIGEN_H

#include <ceres/ceres.h>
#include "jet_extras.h"

template< typename TPixel >
void LinearInitAxis( TPixel x, int size,
                     int *x1, int *x2,
                     TPixel *dx )
{
  const int ix = static_cast<int>(x);
  if (ix < 0) {
    *x1 = 0;
    *x2 = 0;
    *dx = 1.0;
  } else if (ix > size - 2) {
    *x1 = size - 1;
    *x2 = size - 1;
    *dx = 1.0;
  } else {
    *x1 = ix;
    *x2 = ix + 1;
    *dx = *x2 - x;
  }
}

/// Linear interpolation.
template< typename T, class TImage, class TImage2 >
void SampleLinear( const TImage & intensityImage,
                   const TImage2 & intensityGradientX,
                   const TImage2 & intensityGradientY,
                   typename TImage::Scalar y,
                   typename TImage::Scalar x, T* sample )
{
  typedef TImage ImageType;
  typedef typename ImageType::Scalar PixelType;

  int x1, y1, x2, y2;
  PixelType dx, dy;

  // Take the upper left corner as integer pixel positions.
  // XXX halismai (do we need this?)
  //x -= 0.5;
  //y -= 0.5;

  LinearInitAxis(y, intensityImage.rows(), &y1, &y2, &dy);
  LinearInitAxis(x, intensityImage.cols(), &x1, &x2, &dx);

  //Sample intensity
  const T im11 = T(intensityImage(y1, x1));
  const T im12 = T(intensityImage(y1, x2));
  const T im21 = T(intensityImage(y2, x1));
  const T im22 = T(intensityImage(y2, x2));

  sample[0] =(    dy  * ( dx * im11 + (1.0 - dx) * im12 ) +
             (1 - dy) * ( dx * im21 + (1.0 - dx) * im22 ));

  //Sample gradient x
  const T gradx11 = T(intensityGradientX(y1, x1));
  const T gradx12 = T(intensityGradientX(y1, x2));
  const T gradx21 = T(intensityGradientX(y2, x1));
  const T gradx22 = T(intensityGradientX(y2, x2));

  sample[1] =(    dy  * ( dx * gradx11 + (1.0 - dx) * gradx12 ) +
             (1 - dy) * ( dx * gradx21 + (1.0 - dx) * gradx22 ));

  //Sample gradient y
  const T grady11 = T(intensityGradientY(y1, x1));
  const T grady12 = T(intensityGradientY(y1, x2));
  const T grady21 = T(intensityGradientY(y2, x1));
  const T grady22 = T(intensityGradientY(y2, x2));

  sample[2] =(    dy  * ( dx * grady11 + (1.0 - dx) * grady12 ) +
             (1 - dy) * ( dx * grady21 + (1.0 - dx) * grady22 ));
}

// Sample the image at position (x, y) but use the gradient to
// propagate derivatives from x and y. This is needed to integrate the numeric
// image gradients with Ceres's autodiff framework.
template< typename T, class TImage, class TImage2 >
T SampleWithDerivative(const TImage & intensityImage,
                       const TImage2 & intensityGradientX,
                       const TImage2 & intensityGradientY,
                       const T & x,
                       const T & y)
{
  typedef TImage ImageType;
  typedef typename ImageType::Scalar PixelType;

  PixelType scalar_x = ceres::JetOps<T>::GetScalar(x);
  PixelType scalar_y = ceres::JetOps<T>::GetScalar(y);

  PixelType sample[3];
  // Sample intensity image and gradients
  SampleLinear( intensityImage, intensityGradientX, intensityGradientY,
                scalar_y, scalar_x, sample );
  T xy[2] = { x, y };
  return ceres::Chain< PixelType, 2, T >::Rule( sample[0], sample + 1, xy );
}

#endif //

