/**
 * Fast expn implementation.
 */
#include <cassert>
#include <iostream>
#include "util.hh"

#include <opencv/cv.h>
#include <opencv/highgui.h>

void
dumpDoubleImage(double* src,
                int width,
                int height,
                std::string filename)
{
  size_t size = width * height;
  IplImage* img = cvCreateImage (cvSize (width, height),
                                 IPL_DEPTH_8U, 1);
  uchar* data    = (uchar *) img->imageData;

  double norm = 0.;
  for (int i = 0; i < size; ++i)
    norm = max (src[i], norm);

  for (int i = 0; i < size; ++i)
    data[i] = (uchar) ((src[i]/norm)*255);

  std::cout << filename.c_str() << std::endl;
  if(!cvSaveImage(filename.c_str(), img))
    std::cout << "Could not save: " << filename << std::endl;
  cvReleaseImage (&img);
}

void
fast_expn_init (double expn_tab[EXPN_SZ])
{
  DEBUG() << "+fast_expn_init" << std::endl;

  for(int k = 0 ; k < EXPN_SZ + 1 ; ++ k)
    expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ));
  DEBUG() << "-fast_expn_init" << std::endl;
}
