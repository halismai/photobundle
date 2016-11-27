#include "imgproc.h"
#include "types.h"
#include "timer.h"
#include "utils.h"

#include <random>
#include <algorithm>

int main()
{
  const int N = 10;
  const int rows = 480 * 3;
  const int cols = 640 * 3;
  ImageSize im_size(rows, cols);

  const auto image = utils::generateRandomVector<uint8_t>(rows * cols);
  std::vector<float> Ix(image.size());
  std::vector<float> Iy(image.size());

  auto t1 = TimeCode(N, [&]() { imgradient(image.data(), im_size, Ix.data(), Iy.data()); });
  printf("t1: %f\n", t1);

  return 0;
}

