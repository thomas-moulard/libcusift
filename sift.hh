/**
 * SIFT algorithm
 */
#ifndef SIFT_HH_
# define SIFT_HH_
# include <iostream>
# include <list>

# include <cutil.h>

// OpenCV
# include <opencv/cv.h>
# include <opencv/highgui.h>

# include "util.hh"

struct SiftKeypoint
{
  int o;        /* o coordinate (octave) */

  int ix;       /* Integer unnormalized x coordinate. */
  int iy;       /* Integer unnormalized y coordinate. */
  int is;       /* Integer s coordinate. */

  double x;     /* x coordinate. */
  double y;     /* u coordinate. */
  double s;     /* x coordinate. */
  double sigma; /* scale. */
};

struct SiftFeaturePoint
{
  double x;
  double y;
  double scale;
  double angle;
  double desc[128];
};

class Sift
{
public:
  typedef std::list<SiftFeaturePoint> features_t;

  Sift (const IplImage& src_, double pt, double te, double nt,
        int O_, int S_, int o_min_);
  ~Sift ();
  features_t extract ();

protected:
  double fast_expn (double x)
  {
    return ::fast_expn (x, expn_tab_);
  }

  int compute_o_min (int o_min, int w, int h)
  {
    return max ((int) (floor (log2 (min (w, h))) - o_min - 3), 1);
  }

  bool process ();
  bool process_next ();

  double normalize_histogram (double *begin, double *end);
  void compute_keypoint_descriptor(double descr[128], int ind, double angle);

  double*
  get_octave (int s)
  {
    return octave_ + (oW_ * oH_ * (s - s_min));
  }

  void compute_dog ();
  void detect_maxima ();
  void compute_maxima ();
  void refine_maxima ();
  void update_gradient ();
  int compute_keypoint_orientation (int ind, double angles [4]);

  void detect ()
  {
    compute_dog ();
    refine_maxima ();
  }

  void copy_and_upsample_rows (double* dst,
                               const double* src,
                               int width, int height);

  void copy_and_downsample (double* dst,  const double* src,
                            int width, int height, int d);
  void convtransp (double* dst,
                   const double* src,
                   const double* filt,
                   int width, int height, int filt_width);
  void imsmooth(double* dst,
                double* temp,
                double  const* src,
                int width, int height, double sigma);
public:
  int n_keys;
  int n_keys_res;
  SiftKeypoint* keys;

  /// Processed image.
  const IplImage& src;

  const double peak_threshold;
  const double edge_threshold;
  const double norm_threshold;

  const int O;
  const int S;

  const int o_min;
  const int s_min;
  const int s_max;

  /// Image width, heigh and size.
  const int w;
  const int h;
  const int s;

  const int xo;
  const int yo;
  const int so;
private:
  double sigmak_;
  double sigman_;
  double sigma0_;
  double dsigma0_;

  int oCur_;
  int oW_;
  int oH_;
  int oGrad_;

  double* octave_;
  double* dog_;
  double* gradient_;
  double* tmp_;
  double* im_;

  int filt_width_;
  double filt_sigma_;
  int filt_res_;
  double* filt_;

  double expn_tab_[EXPN_SZ];
};

#endif //! SIFT_HH_
