#include <immintrin.h>
#include "utils.h"
#include "imgproc.h"
#include "timer.h"

#include <smmintrin.h>

void disparityToDepth2(const float* d_ptr, int N, float Bf, float* z_ptr)
{
  auto BF = _mm_set1_ps(Bf);
  auto MD = _mm_set1_ps(0.01f);
  auto IV = _mm_set1_ps(-1.0f);

#define USE_RCP 1

  int i = 0;
  for(i = 0; i <= N - 4; i += 4) {
    auto d = _mm_load_ps( d_ptr + i );
    auto m = _mm_cmpgt_ps(d, MD);
#if USE_RCP
    auto z = _mm_div_ps(BF, d);
#else
    auto z = _mm_mul_ps(BF, _mm_rcp_ps(d));
#endif
    auto zz = _mm_blendv_ps(IV, z, m);
    _mm_store_ps(z_ptr + i, zz);
  }

#undef USE_RCP

  for( ; i < N; ++i) {
    z_ptr[i] = d_ptr[i] > 0.01f ? (Bf / d_ptr[i]) : -1.0f;
  }
}

int main()
{
  int rows = 640 * 480 * 8, cols = 1;
  int N = rows * cols;
  auto dmap = utils::generateRandomVector<float>(N + 1);
  float bf = 0.12 * 500;

  dmap[2] = -1.0f;

  std::vector<float> z1(dmap.size());
  disparityToDepth(dmap.data(), ImageSize(rows, cols), bf, z1.data());
  {
    auto t = TimeCode(100, disparityToDepth, dmap.data(), ImageSize(rows, cols), bf, z1.data());
    printf("time %g\n", t);
  }

  std::vector<float> z2(dmap.size());
  disparityToDepth2(dmap.data(), N, bf, z2.data());
  {
    auto t = TimeCode(100, disparityToDepth2, dmap.data(), N, bf, z1.data());
    printf("time %g\n", t);
  }

  for(size_t i = 0; i < z1.size(); ++i) {
    auto err = std::abs(z1[i] - z2[i]);
    if(err > 1e-3) {
      printf("bad %zu: %g [%f %f]\n", i, err, z1[i], z2[i]);
    }
  }

  return 0;
}


