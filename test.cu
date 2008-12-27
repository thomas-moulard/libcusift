/**
 * Main function.
 */
#include "sift.hh"
#include "util.hh"

char* inFilename = "data/left2.pgm";
char* outFilename = "res.bmp";


IplImage*
computeResultImage (const std::list<SiftFeaturePoint>& sift, IplImage& img,
                    double z=1., double xOff=0., double yOff=0.)
{
  typedef std::list<SiftFeaturePoint>::const_iterator citer_t;
  IplImage* res = cvCloneImage (&img);

  for (citer_t it = sift.begin ();
       it != sift.end(); ++it)
    {
      CvPoint o = cvPoint ((int)((it->x+xOff)*z), (int)((it->y+yOff)*z));
      CvPoint o2 = cvPoint ((int)((it->x+xOff)*z*cos(it->angle)),
                            (int)((it->y+yOff)*z*sin(it->angle)));
      cvCircle(&img, o,
               (int)((it->scale+1)*z), cvScalar(255,0,0), 1);
      cvLine(&img, o, o2, cvScalar(255,0,0), 1);
    }
  return res;
}


void
testSift (int argc, char** argv)
{
  CUT_DEVICE_INIT (argc, argv);

  fast_expn_init ();

  // Load image
  IplImage* img = 0;
  img = cvLoadImage (inFilename);
  if(!img) {
    std::cout << "Could not load image file: "
              << inFilename << std::endl;
    exit (0);
  }

  // Convert to greyscale.
  IplImage* greyimg = cvCreateImage (cvSize (img->width, img->height),
                                     IPL_DEPTH_8U, 1);
  cvCvtColor(img, greyimg, CV_BGR2GRAY);

  // Init timer.
  unsigned int timer = 0;
  CUT_SAFE_CALL (cutCreateTimer (&timer));
  CUT_SAFE_CALL(cutStartTimer (timer));

  // Run sift and store image result.
  Sift sift (*img, 5.2, 5.2, 5.2, 8, 5, 1);
  std::cout << "Begin SIFT extraction." << std::endl;
  std::list<SiftFeaturePoint> fps = sift.extract ();
  std::cout << "SIFT extraction done." << std::endl;
  std::cout << fps.size() << " extracted feature(s)." << std::endl;
  IplImage* res = computeResultImage (fps, *greyimg);

  // Write result image.
  if(!cvSaveImage(outFilename, res))
    std::cout << "Could not save: " << outFilename << std::endl;
  else
    std::cout << outFilename << " succesfully written." << std::endl;

    // Release memory.
    cvReleaseImage (&res);
    cvReleaseImage (&greyimg);
    cvReleaseImage (&img);
}

// Program main
int
main (int argc, char** argv)
{
    testSift (argc, argv);
    CUT_EXIT (argc, argv);
}
