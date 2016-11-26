#include "imgproc.h"
#include "types.h"
#include "timer.h"

#include <random>
#include <algorithm>

int main()
{
  const int N = 100;
  const int rows = 480 * 4;
  const int cols = 640 * 4;
  ImageSize im_size(rows, cols);

  std::vector<uint8_t> image(rows * cols);
  {
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    std::mt19937 rng;
    auto gen = std::bind(dist, rng);
    std::generate(std::begin(image), std::end(image), gen);
  }

  std::vector<float> Ix(image.size());
  std::vector<float> Iy(image.size());

  auto t1 = TimeCode(N, imgradient, image.data(), im_size, Ix.data(), Iy.data());
  printf("t1: %f\n", t1);

  auto t2 = TimeCode(N, imgradientParallel, image.data(), im_size, Ix.data(), Iy.data());
  printf("t2: %f\n", t2);

  return 0;
}

