/**
 * Main function.
 */
#include "sift.hh"
#include "util.hh"

char* inFilename = "data/left2.pgm";
char* outFilename = "res.bmp";


void
computeResultImage (const std::list<SiftFeaturePoint>& sift, IplImage& img,
                    double z=1., double xOff=0., double yOff=0.)
{
  std::cout << "+computeResultImage" << std::endl;
  typedef std::list<SiftFeaturePoint>::const_iterator citer_t;

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
  std::cout << "-computeResultImage" << std::endl;
}


void
testSift (int argc, char** argv)
{
  CUT_DEVICE_INIT (argc, argv);

  // Load image
  IplImage* img = 0;
  img = cvLoadImage (inFilename);
  if(!img)
    {
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


  // Run sift and store image result.
  {
    Sift sift (*greyimg, 1., 1., 1., 4, 3, 1);
    std::cout << "Begin SIFT extraction." << std::endl;

    CUT_SAFE_CALL(cutStartTimer (timer));
    std::list<SiftFeaturePoint> fps = sift.extract ();
    CUT_SAFE_CALL (cutStopTimer (timer));

    std::cout << "SIFT extraction done." << std::endl;
    std::cout << fps.size() << " extracted feature(s)." << std::endl;
    computeResultImage (fps, *greyimg);
  }

  // Write result image.
  if(!cvSaveImage(outFilename, greyimg))
    std::cout << "Could not save: " << outFilename << std::endl;
  else
    std::cout << outFilename << " succesfully written." << std::endl;

  std::cout << "Processing time: " << cutGetTimerValue (timer)
            << " (ms)" << std::endl;
  CUT_SAFE_CALL (cutDeleteTimer (timer));

  // Release memory.
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
