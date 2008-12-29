/**
 * Main function.
 */
#include "sift.hh"
#include "util.hh"

char* inFilename = "data/IMG_1011.jpg";
char* outFilename = "res.bmp";


void
computeResultImage (const std::list<SiftFeaturePoint>& sift, IplImage& img,
                    double z=1., double xOff=0., double yOff=0.)
{
  DEBUG() << "+computeResultImage" << std::endl;
  typedef std::list<SiftFeaturePoint>::const_iterator citer_t;

  for (citer_t it = sift.begin ();
       it != sift.end(); ++it)
    {
      int r = (int)((it->scale+1)*z);
      CvPoint o = cvPoint ((int)((it->x+xOff)*z), (int)((it->y+yOff)*z));
      CvPoint o2 = cvPoint ((int)((it->x+xOff)+(r*cos(it->angle))),
                            (int)((it->y+yOff)+(r*sin(it->angle))));
      cvCircle(&img, o, r, cvScalar(0, 0, 255), 1);
      cvLine(&img, o, o2, cvScalar(0, 0, 255), 1);
    }
  DEBUG() << "-computeResultImage" << std::endl;
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
    // img, peak th, edge th, norm th, O, S, o_min
    Sift sift (*greyimg, 1., 10., 0., 4, 3, 1);
    std::cout << "Begin SIFT extraction." << std::endl;

    CUT_SAFE_CALL(cutStartTimer (timer));
    std::list<SiftFeaturePoint> fps = sift.extract ();
    CUT_SAFE_CALL (cutStopTimer (timer));

    std::cout << "SIFT extraction done." << std::endl;
    std::cout << fps.size() << " extracted feature(s)." << std::endl;
    computeResultImage (fps, *img);
  }

  // Write result image.
  if(!cvSaveImage(outFilename, img))
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
