#include <opencv2/highgui/highgui.hpp>
#include "debug.h"
#include "imgproc.h"

#include <cstdio>

void writeToFile(const char* fn, const float* p, int rows, int cols)
{
  FILE* fp = fopen(fn, "w");
  if(!fp)
    Fatal("Failed to open file\n");

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < cols; ++j) {
      fprintf(fp, "%g ", *p++);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}

int main()
{
  auto I = cv::imread("/home/halismai/lena_gray.png", cv::IMREAD_GRAYSCALE);

  float* Ix = new float[I.rows * I.cols];
  float* Iy = new float[I.rows * I.cols];

  imgradient(I.ptr<uint8_t>(), ImageSize(I.rows, I.cols), Ix, Iy);


  writeToFile("Ix", Ix, I.rows, I.cols);
  writeToFile("Iy", Iy, I.rows, I.cols);

  return 0;
}
