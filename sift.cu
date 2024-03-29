/**
 * SIFT implementation.
 */

#include <sstream>
#include "sift.hh"

#ifndef NDEBUG
# define DEBUG_DUMP_IMAGE(PREFIX, SRC)                                  \
  for (int s = s_min; s <= s_max - 1; ++s)                              \
    {                                                                   \
      std::ostringstream ss;                                            \
      ss << (PREFIX) << "." << oCur_ << "." << s << ".bmp";             \
      dumpDoubleImage ((SRC) + (s-s_min) * oW_ * oH_, oW_, oH_, ss.str()); \
    }
#else
# define DEBUG_DUMP_IMAGE(PREFIX, SRC)
#endif //! NDEBUG

Sift::Sift (const IplImage& src_, double pt, double et, double nt,
            int O_, int S_, int o_min_)
  : n_keys (0),
    n_keys_res (0),
    keys (0),
    src (src_),
    peak_threshold (pt),
    edge_threshold (et),
    norm_threshold (nt),
    O ((O_ < 0) ? compute_o_min (o_min_, src_.width, src_.height) : O_),
    S (S_),
    o_min (o_min_),
    s_min (-1),
    s_max (S_ + 1),
    w (src.width),
    h (src.height),
    s (shift_left (w, -o_min) * shift_left (h, -o_min) * sizeof (double)),
    sigmak_ (pow(2.0, 1.0 / S_)), // 2^(1/S)
    sigman_ (.5),
    sigma0_ (1.6 * sigmak_),
    dsigma0_ (sigma0_ * sqrt (1.0 - 1.0/(sigmak_*sigmak_))), // sigma0 * sqrt(1 - 1/sigmak²)
    oCur_ (o_min_),
    oW_ (O),
    oH_ (0),
    oGrad_ (o_min_ - 1),
    octave_ (h_malloc<double> (s*(s_max-s_min+1))),
    dog_ (h_malloc<double> (s*(s_max-s_min))),
    gradient_ (h_malloc<double> (s*2*(s_max-s_min))),
    tmp_ (h_malloc<double> (s)),
    im_ (h_malloc<double> (src_.width * src.height * sizeof (double))),
    filt_width_ (-1),
    filt_sigma_ (-1.),
    filt_res_ (-1),
    filt_ (0),
    expn_tab_ ()
{
  DEBUG() << "Create SIFT filter with parameter:" << std::endl
          << "* Number of octaves: " << O << std::endl
          << "* Min octave: " << o_min << std::endl
          << "* S: " << S << std::endl
          << "* Peak/Edge/Norm thresholds: " << pt << "/" << et << "/" << nt
          << std::endl;
  DEBUG() << (s/sizeof (double))
          << " / "
          << s << " / "
          << (s*(s_max-s_min+1)) << " / "
          << (s*(s_max-s_min)) << " / "
          << (src_.width * src.height * sizeof (double)) << std::endl;

  fast_expn_init (expn_tab_);

  // Convert openCV image to double*
  int offset = 0;
  for (int y = 0; y < src.height; ++y)
    for (int x = 0; x < src.width; ++x)
      im_[offset++] = cvGet2D(&src, y, x).val[0];
}

Sift::~Sift ()
{
  DEBUG() << "+Sift::~Sift ()" << std::endl;

  DEBUG() << "Free keys" << std::endl;
  if (keys)
    h_free<SiftKeypoint> (keys);

  DEBUG() << "Free image (double)" << std::endl;
  h_free<double> (im_);

  DEBUG() << "Free temp buffer" << std::endl;
  h_free<double> (tmp_);
  DEBUG() << "Free difference of gaussians" << std::endl;
  h_free<double> (dog_);
  DEBUG() << "Free gradient data" << std::endl;
  h_free<double> (gradient_);
  DEBUG() << "Free octave" << std::endl;
  h_free<double> (octave_);

  DEBUG() << "Free filter buffer" << std::endl;
  if (!!filt_)
    h_free<double> (filt_);
  DEBUG() << "-Sift::~Sift ()" << std::endl;
}

bool
Sift::process ()
{
  DEBUG() << "+Process" << std::endl;
  if (!O)
    return false;

  oCur_ = o_min;
  n_keys = 0;
  oW_ = shift_left (w, -oCur_);
  oH_ = shift_left (h, -oCur_);

  if (this->O == 0)
    return false;

  double* octave = get_octave (s_min);

  if (o_min < 0)
    {
      copy_and_upsample_rows (tmp_, im_, w, h);
      copy_and_upsample_rows (octave, tmp_, h, 2 * w);
      for (int o = -1; o > o_min; --o)
        {
          copy_and_upsample_rows (tmp_, octave,
                                    w << -o,      h << -o );
          copy_and_upsample_rows (octave, tmp_,
                                  w << -o, 2 * (h << -o));
        }
    }
  else if (o_min > 0)
    copy_and_downsample (octave, im_, w, h, o_min);
  else
    memcpy(octave, im_, w*h*sizeof (double));

  double sa = sigma0_ * pow (sigmak_, s_min);
  double sb = sigman_ * pow (2.0, -o_min);

  if (sa > sb)
    {
      double sd = sqrt (sa*sa - sb*sb);
      DEBUG() << "sa > sb => " << sd << std::endl;
      imsmooth (octave, tmp_, octave, oW_, oH_, sd);
    }

  // compute octave.
  for (int s = s_min + 1; s <= s_max; ++s)
    {
      DEBUG() << "compute octave " << s << std::endl;
      double sd = dsigma0_ * pow (sigmak_, s);
      imsmooth (get_octave(s), tmp_,
                get_octave(s - 1), oW_, oH_, sd);
    }
  DEBUG_DUMP_IMAGE("octave", octave_);
  DEBUG() << "-Process" << std::endl;
  return true;
}

bool
Sift::process_next ()
{
  DEBUG() << "+Process next" << std::endl;
  if (oCur_ == o_min + O - 1)
    return false;

  int s_best = min (s_min + S, s_max);
  double* pt = get_octave (s_best);
  double* octave = get_octave (s_min);

  copy_and_downsample (octave, pt, oW_, oH_, 1);

  oCur_ += 1, n_keys = 0;
  oW_ = shift_left (w, -oCur_);
  oH_ = shift_left (h, -oCur_);

  double sa = sigma0_ * powf (sigmak_, s_min);
  double sb = sigma0_ * powf (sigmak_, s_best - S);

  if (sa > sb)
    {
      double sd = sqrt (sa*sa - sb*sb);
      imsmooth (octave, tmp_, octave, oW_, oH_, sd);
    }

  for(int s = s_min + 1 ; s <= s_max ; ++s)
    {
      double sd = dsigma0_ * pow (sigmak_, s);
      imsmooth (get_octave (s), tmp_,
                get_octave (s - 1), oW_, oH_, sd);
    }

  DEBUG_DUMP_IMAGE("octave", octave_);
  DEBUG() << "-Process next" << std::endl;
  return true;
}

#define SINGLE_EPSILON 1.19209290E-07F
double
Sift::normalize_histogram (double *begin, double *end)
{
  double norm = .0;
  for (double* iter = begin ; iter != end ; ++iter)
    norm += (*iter) * (*iter);
  norm = sqrt (norm) + SINGLE_EPSILON;
  for (double* iter = begin; iter != end ; ++iter)
    *iter /= norm;
  return norm;
}
#undef SINGLE_EPSILON

#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)
void
Sift::compute_keypoint_descriptor(double descr[128], int ind, double angle)
{
  assert (ind < n_keys);
  SiftKeypoint* k = keys + ind;

  const double magnif = 3.0;
  const int NBO = 8;
  const int NBP = 4;
  double xper = pow (2.0, oCur_);
  const int xo = 2;
  const int yo = 2 * oW_;
  const int so = 2 * oW_ * oH_;
  double x = k->x / xper;
  double y = k->y / xper;
  double sigma = k->sigma / xper;

  int xi = (int) (x + .5);
  int yi = (int) (y + .5);
  int si = k->is;

  const double st0 = sin (angle);
  const double ct0 = cos (angle);
  const double SBP = magnif * sigma;
  const int W = (int)floor (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + .5);

  const int binto = 1;
  const int binyo = NBO * NBP;
  const int binxo = NBO;

  if(k->o  != oCur_ || xi <  0 || xi >= oW_ || yi <  0 ||
     yi >= oH_ - 1 || si < s_min + 1 || si > s_max - 2)
    return;

  update_gradient ();

  memset (descr, 0, sizeof(double) * NBO*NBP*NBP);

  const double* pt  = gradient_ + xi*xo + yi*yo + (si - s_min - 1)*so;
  double* dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo;

  for(int dyi =  max (-W, 1 - yi); dyi <= min (W, oH_ - yi - 2); ++dyi)
    for(int dxi =  max (-W, 1 - xi); dxi <= min (W, oW_ - xi - 2); ++ dxi)
      {
        double mod   = *(pt + dxi*xo + dyi*yo + 0);
        double angle_ = *(pt + dxi*xo + dyi*yo + 1);
        double theta = mod_2pi (angle_ - angle);

        double dx = xi + dxi - x;
        double dy = yi + dyi - y;

        double nx = (ct0 * dx + st0 * dy) / SBP;
        double ny = (-st0 * dx + ct0 * dy) / SBP;
        double nt = NBO * theta / (2 * M_PI);

        double const wsigma = NBP/2;
        double win = fast_expn
          ((nx*nx + ny*ny)/(2.0 * wsigma * wsigma));

        int binx = (int)floor (nx - .5);
        int biny = (int)floor (ny - .5);
        int bint = (int)floor (nt);
        double rbinx = nx - (binx + .5);
        double rbiny = ny - (biny + .5);
        double rbint = nt - bint;

        for(int dbinx = 0; dbinx < 2; ++dbinx)
          for(int dbiny = 0; dbiny < 2; ++dbiny)
            for(int dbint = 0; dbint < 2; ++dbint)
              {
                if (binx + dbinx >= - (NBP/2) &&
                    binx + dbinx <    (NBP/2) &&
                    biny + dbiny >= - (NBP/2) &&
                    biny + dbiny <    (NBP/2) )
                  {
                    double weight = win
                      * mod
                      * abs (1 - dbinx - rbinx)
                      * abs (1 - dbiny - rbiny)
                      * abs (1 - dbint - rbint);
                    atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight;
                  }
              }
      }

  double norm = normalize_histogram (descr, descr + NBO*NBP*NBP);

  if(norm_threshold && norm <  norm_threshold)
    for (int bin = 0; bin < NBO*NBP*NBP; ++ bin)
      descr [bin] = 0;
  else
    {
      for(int bin = 0; bin < NBO*NBP*NBP; ++ bin)
        if (descr [bin] > 0.2) descr [bin] = 0.2;

      normalize_histogram (descr, descr + NBO*NBP*NBP);
    }
}

Sift::features_t
Sift::extract ()
{
  features_t res;

  if (!process ())
    return res;
  while (true)
    {
      detect ();

      DEBUG() << n_keys << " generated keys." << std::endl;
      for (int i=0; i<n_keys; ++i)
        {
          double angles[4];
          memset (angles, 0, sizeof(double) * 4);
          int n_angles = compute_keypoint_orientation (i, angles);
          for (int q=0; q < n_angles; ++q)
            {
              double descr[128];
              compute_keypoint_descriptor(descr, i, angles [q]);

              SiftFeaturePoint fp;
              fp.x = keys[i].x, fp.y = keys[i].y;
              fp.scale = keys[i].sigma;
              fp.angle = angles[q];
              for (int k = 0; k < 128; ++k)
                fp.desc[k] = 512 * descr[k];
              res.push_back (fp);
            }
        }

      // build SIFT descriptors.
      if (!process_next ())
        return res;
    }
}

// Parallel version of difference of gaussian.
//
// blockId == row number
// threadId == scale
__global__ void
compute_dog_row (double* dev_dog, double* dev_oct, int w, int h, int s_min)
{
  const int s  = s_min + threadIdx.x;
  double* pt = dev_dog + (s-s_min)*w*h + w*blockIdx.x;
  double* src_a = dev_oct + (s-s_min)*w*h + w*blockIdx.x;
  double* src_b = dev_oct + (s-s_min+1)*w*h + w*blockIdx.x;
  double* end_a = src_a + w;
  while (src_a != end_a)
    *pt++ = *src_b++ - *src_a++;
}

void
Sift::compute_dog ()
{
  DEBUG() << "+compute_dog" << std::endl;

  double* dev_dog = d_malloc<double> (s*(s_max-s_min));
  double* dev_oct = d_malloc<double> (s*(s_max-s_min+1));

  cudaMemcpy (dev_oct, octave_, s*(s_max-s_min+1), cudaMemcpyHostToDevice);
  compute_dog_row<<<oH_, s_max - s_min>>> (dev_dog, dev_oct, oW_, oH_, s_min);
  cudaMemcpy (dog_, dev_dog, s*(s_max-s_min), cudaMemcpyDeviceToHost);

  d_free (dev_oct);
  d_free (dev_dog);

  DEBUG_DUMP_IMAGE("dog", dog_);
  DEBUG() << "-compute_dog" << std::endl;
}




#define CHECK_NEIGHBORS(CMP,SGN)                \
  ( v CMP ## = SGN 0.8 * peak_threshold &&      \
    v CMP *(pt + xo) &&                         \
    v CMP *(pt - xo) &&                         \
    v CMP *(pt + so) &&                         \
    v CMP *(pt - so) &&                         \
    v CMP *(pt + yo) &&                         \
    v CMP *(pt - yo) &&                         \
                                                \
    v CMP *(pt + yo + xo) &&                    \
    v CMP *(pt + yo - xo) &&                    \
    v CMP *(pt - yo + xo) &&                    \
    v CMP *(pt - yo - xo) &&                    \
                                                \
    v CMP *(pt + xo      + so) &&               \
    v CMP *(pt - xo      + so) &&               \
    v CMP *(pt + yo      + so) &&               \
    v CMP *(pt - yo      + so) &&               \
    v CMP *(pt + yo + xo + so) &&               \
    v CMP *(pt + yo - xo + so) &&               \
    v CMP *(pt - yo + xo + so) &&               \
    v CMP *(pt - yo - xo + so) &&               \
                                                \
    v CMP *(pt + xo      - so) &&               \
    v CMP *(pt - xo      - so) &&               \
    v CMP *(pt + yo      - so) &&               \
    v CMP *(pt - yo      - so) &&               \
    v CMP *(pt + yo + xo - so) &&               \
    v CMP *(pt + yo - xo - so) &&               \
    v CMP *(pt - yo + xo - so) &&               \
    v CMP *(pt - yo - xo - so) )

void
Sift::detect_maxima ()
{
  DEBUG() << "+Detect maxima" << std::endl;
  const int xo = 1;
  const int yo = oW_;
  const int so = oW_ * oH_;
  double* pt  = dog_ + xo + yo + so;

  for (int s = s_min + 1; s <= s_max - 2; ++s)
    {
      for(int y = 1; y < oH_ - 1; ++y)
        {
          for(int x = 1; x < oW_ - 1; ++x)
            {
              double v = *pt;
              if (CHECK_NEIGHBORS(>,+) ||
                  CHECK_NEIGHBORS(<,-) )
                {
                  // Allocate keypoints.
                  if (n_keys >= n_keys_res)
                    {
                      n_keys_res += 500;
                      if (keys)
                        keys = (SiftKeypoint*) realloc (keys,
                                                        n_keys_res *
                                                        sizeof(SiftKeypoint));
                      else
                        keys = h_malloc<SiftKeypoint> (n_keys_res *
                                                       sizeof(SiftKeypoint));
                    }

                  SiftKeypoint* k = keys + (n_keys ++);
                  k->ix = x, k->iy = y, k->is = s;
                }
              pt += 1;
            }
          pt += 2;
        }
      pt += 2 * yo;
    }
  DEBUG() << "-Detect maxima (" << n_keys << " points detected)." << std::endl;
}

#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))
#define Aat(i,j)     (A[(i)+(j)*3])
void
Sift::refine_maxima ()
{
  DEBUG() << "+Refine maxima" << std::endl;
  SiftKeypoint* k = keys;

  double maxa = 0;
  double maxabsa = 0;
  int maxi  = -1;
  double tmp = 0.;
  double* pt = 0;
  double xper  = pow (2.0, oCur_);

  // Offset
  const int xo = 1;
  const int yo = oW_;
  const int so = oW_ * oH_;

  for (int i = 0; i < n_keys; ++i)
    {
      int x =  keys[i].ix;
      int y =  keys[i].iy;
      int s =  keys[i].is;

      double Dx=0, Dy=0, Ds=0,
        Dxx=0, Dyy=0, Dss=0,
        Dxy=0, Dxs=0, Dys=0;
      double A [3*3], b [3];
      int dx = 0, dy = 0;

      for (int iter = 0; iter < 5; ++iter)
        {
          x += dx, y += dy;

          pt = dog_ + xo * x + yo * y + so * (s - s_min);

          // Hessian
          Dx = .5 * (at(+1,0,0) - at(-1,0,0));
          Dy = .5 * (at(0,+1,0) - at(0,-1,0));
          Ds = .5 * (at(0,0,+1) - at(0,0,-1));

          Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0));
          Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0));
          Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0));

          Dxy = 0.25 * ( at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0) );
          Dxs = 0.25 * ( at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1) );
          Dys = 0.25 * ( at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1) );

          Aat(0,0) = Dxx, Aat(1,1) = Dyy, Aat(2,2) = Dss,
            Aat(0,1) = Aat(1,0) = Dxy, Aat(0,2) = Aat(2,0) = Dxs,
            Aat(1,2) = Aat(2,1) = Dys;

          b[0] = - Dx, b[1] = - Dy, b[2] = - Ds;
          // Gauss
          for(int j = 0; j < 3; ++j)
            {
              for (int i = j; i < 3; ++i)
                {
                  double a = Aat (i,j), absa = abs (a);
                  if (absa > maxabsa)
                    maxa = a, maxabsa = absa, maxi = i;
                }

              // Singular.
              if (maxabsa < 1e-10f)
                {
                  b[0] = 0, b[1] = 0, b[2] = 0;
                  break;
                }

              for(int jj = j; jj < 3; ++jj)
                {
                  tmp = Aat(maxi,jj);
                  Aat(maxi,jj) = Aat(j,jj);
                  Aat(j,jj) = tmp;
                  Aat(j,jj) /= maxa;
                }
              tmp = b[j]; b[j] = b[maxi]; b[maxi] = tmp;
              b[j] /= maxa;

              for (int ii = j+1; ii < 3; ++ii)
                {
                  double x = Aat(ii,j);
                  for (int jj = j; jj < 3; ++jj)
                    Aat(ii,jj) -= x * Aat(j,jj);
                  b[ii] -= x * b[j];
                }
            }

          for (int i = 2; i > 0; --i)
            {
              double x = b[i];
              for (int ii = i-1; ii >= 0; --ii)
                b[ii] -= x * Aat(ii,i);
            }

          dx= ((b[0] >  0.6 && x < oW_ - 2) ?  1 : 0)
            + ((b[0] < -0.6 && x > 1)       ? -1 : 0);

          dy= ((b[1] >  0.6 && y < oH_ - 2) ?  1 : 0)
            + ((b[1] < -0.6 && y > 1)       ? -1 : 0);

          if (dx == 0 && dy == 0)
            break;
        }

      {
        double val   = at(0,0,0)
          + .5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
        double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy);
        double xn = x + b[0];
        double yn = y + b[1];
        double sn = s + b[2];

        bool good =
          abs (val) > peak_threshold            &&
          score < (edge_threshold+1)*(edge_threshold+1)/edge_threshold &&
          score           >= 0                  &&
          abs (b[0]) <  1.5                     &&
          abs (b[1]) <  1.5                     &&
          abs (b[2]) <  1.5                     &&
          xn              >= 0                  &&
          xn              <= oW_ - 1            &&
          yn              >= 0                  &&
          yn              <= oH_ - 1            &&
          sn              >= s_min              &&
          sn              <= s_max;

        if (good)
          {
            k->o = oCur_;
            k->ix = x, k->iy = y, k->is = s;
            k->s = sn;
            k->x = xn * xper, k->y = yn * xper;
            k->sigma = sigma0_ * pow (2.0, sn/S) * xper;
            ++k;
          }
      }
    }

  assert (k >= keys);
  n_keys = k - keys;
  DEBUG() << "-Refine maxima (" << n_keys << ")" << std::endl;
}
#undef at

#define SAVE_BACK                                  \
  *grad++ = sqrt (gx*gx + gy*gy);                  \
  *grad++ = mod_2pi(atan2 (gy, gx) + 2*M_PI);      \
  ++src


void
Sift::update_gradient ()
{
  const int xo = 1;
  const int yo = oW_;
  const int so = oW_ * oH_;

  if (oGrad_ == oCur_)
    return;

  for (int s  = s_min + 1;
       s <= s_max - 2; ++ s)
    {
      double* src;
      double* end;
      double gx, gy;
      double* grad = gradient_ + 2 * so * (s - s_min -1);
      src  = get_octave (s);

      gx = src[xo] - src[0];
      gy = src[yo] - src[0];
      SAVE_BACK;

      end = (src - 1) + oW_ - 1;
      while (src < end)
        {
          gx = .5 * (src[xo] - src[-xo]);
          gy =        src[yo] - src[0];
          SAVE_BACK;
        }

      gx = src[0]   - src[-xo];
      gy = src[yo] - src[0];
      SAVE_BACK;

      for (int y = 1; y < oH_ -1; ++y)
        {
          gx = src[xo] - src[0];
          gy = .5 * (src[yo] - src[-yo]);
          SAVE_BACK;

          end = (src - 1) + oW_ - 1;
          while (src < end)
            {
              gx = .5 * (src[xo] - src[-xo]);
              gy = .5 * (src[yo] - src[-yo]);
              SAVE_BACK;
            }

          gx = src[0] - src[-xo];
          gy = .5 * (src[yo] - src[-yo]);
          SAVE_BACK;
        }

      gx = src[xo] - src[0];
      gy = src[0] - src[-yo];
      SAVE_BACK;

      end = (src - 1) + oW_ - 1;
      while (src < end)
        {
          gx = .5 * (src[xo] - src[-xo]);
          gy = src[0] - src[-yo];
          SAVE_BACK;
        }
      gx = src[0] - src[-xo];
      gy = src[0] - src[-yo];
      SAVE_BACK;
    }
  oGrad_ = oCur_;
}

#define at(dx,dy) (*(pt + xo * (dx) + yo * (dy)))
int
Sift::compute_keypoint_orientation (int ind, double angles [4])
{
  assert (ind < n_keys);
  SiftKeypoint* k = keys + ind;

  const double winf = 1.5;
  double xper = pow (2.0, oCur_);

  // Offsets
  const int xo = 2;
  const int yo = 2 * oW_;
  const int so = 2 * oW_ * oH_;
  double x = k->x/xper, y = k->y/xper;
  double sigma = k->sigma/xper;

  int xi = (int) (x + .5),
    yi = (int) (y + .5), si = k->is;

  const double sigmaw = winf * sigma;
  int W = (int)max (floor (3.0 * sigmaw), 1.);
  const int nbins = 36;
  double hist [nbins], maxh;

  if(k->o != oCur_)
    return 0;

  // Check bounds.
  if(xi < 0 || xi > oW_ - 1 || yi < 0 || yi > oH_ - 1 ||
     si < s_min + 1 || si > s_max - 2 )
    return 0;

  update_gradient ();

  memset (hist, 0, sizeof(double) * nbins);

  double* pt =  gradient_ + xo*xi + yo*yi + so*(si - s_min - 1);

  for (int ys  =  max (-W, -yi);
       ys <=  min (W, oH_ - 1 - yi); ++ys)
    for (int xs  = max (-W, -xi);
         xs <= min (W, oW_ - 1 - xi); ++xs)
      {
        double dx = (double)(xi + xs) - x;
        double dy = (double)(yi + ys) - y;
        double r2 = dx*dx + dy*dy;
        double wgt, mod, ang, fbin;

        if (r2 >= W*W + 0.6)
          continue;

        wgt  = fast_expn (r2 / (2*sigmaw*sigmaw));
        mod  = *(pt + xs*xo + ys*yo    );
        ang  = *(pt + xs*xo + ys*yo + 1);
        fbin = nbins * ang / (2 * M_PI);

        int    bin  = (int)floor (fbin - .5);
        double rbin = fbin - bin - .5;
        hist [(bin + nbins) % nbins] += (1 - rbin) * mod * wgt;
        hist [(bin + 1    ) % nbins] += (rbin) * mod * wgt;
      }

  for (int iter = 0; iter < 6; iter++)
    {
      double prev  = hist [nbins - 1];
      double first = hist [0];
      int i = 0;
      for (; i < nbins - 1; i++)
        {
          double newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
          prev = hist[i];
          hist[i] = newh;
        }
      hist[i] = (prev + hist[i] + first) / 3.0;
    }

  maxh = 0;
  for (int i = 0; i < nbins; ++i)
    maxh = max (maxh, hist [i]);

  int n_angles = 0;
  for (int i = 0; i < nbins; ++i)
    {
      double h0 = hist [i];
      double hm = hist [(i - 1 + nbins) % nbins];
      double hp = hist [(i + 1 + nbins) % nbins];

      if (h0 > .8 * maxh && h0 > hm && h0 > hp)
        {
          double di = - .5 * (hp - hm) / (hp + hm - 2 * h0);
          double th = 2 * M_PI * (i + di + .5) / nbins;
          angles [n_angles++] = th;
          if (n_angles == 4)
            return n_angles;
        }
    }
  return n_angles;
}
#undef at

void
Sift::copy_and_upsample_rows (double* dst,
                             const double* src,
                             int width, int height)
{
  DEBUG() << "+copy_and_upsample_rows" << std::endl;

  for (int y = 0; y < height; ++y)
    {
      double a = *src++;
      double b = a;

      for (int x = 0; x < width - 1; ++x)
        {
          b = *src++;
          *dst = a; dst += height;
          *dst = .5 * (a + b); dst += height;
          a = b;
        }
      *dst = b; dst += height;
      *dst = b; dst += height;
      dst += 1 - width * 2 * height;
    }
  DEBUG() << "-copy_and_upsample_rows" << std::endl;
}

/*
 * Parallelized version of copy_and_downsample.
 * block == row
 * thread == 1/32nth of a row
 */
__global__ void
_copy_and_downsample (double* dst, const double* src,
                           int width, int height, int d)
{
  const int y = d * blockIdx.x;
  const int line = width-d+1;
  const int start = threadIdx.x*line/blockDim.x;
  const int end = start + (line/blockDim.x);
  const double* srcrowp = src + y * width + start;
  dst += (width-d+2)/d*y/d + start/d;

  for (int x = start; x < end; x+=d)
    {
      *dst++ = *srcrowp;
      srcrowp += d;
    }
}


void
Sift::copy_and_downsample (double* dst,  const double* src,
                           int width, int height, int d)
{
  DEBUG() << "+copy_and_downsample" << std::endl;

  d = 1 << d; // d = 2^d
  int size_src = width*height*sizeof (double);
  int size_dst = (int) (floor ((width+1)/d)*floor ((height+1)/d)*sizeof (double));
  double* dev_dst = d_malloc<double> (size_dst);
  double* dev_src = d_malloc<double> (size_src);

  cudaMemcpy (dev_src, src, size_src, cudaMemcpyHostToDevice);
  _copy_and_downsample <<<height/d, 1>>> (dev_dst, dev_src, width, height, d);
   cudaMemcpy (dst, dev_dst, size_dst, cudaMemcpyDeviceToHost);

  d_free (dev_dst);
  d_free (dev_src);

  DEBUG() << "-copy_and_downsample" << std::endl;
}

__global__ void
_convtransp (double* dst, const double* src, const double* filt,
             int width, int height, int filt_width)
{
  int j = blockIdx.x;
  src += j * width;
  dst += j * width * height - (j * (width*height - 1));

  int part = width/blockDim.x;
  int start = threadIdx.x * part;
  int end = start + part;

  for(int i = start; i < end; ++i)
    {
      double acc = .0;
      const double *g = filt;
      const double *start = src + (i - filt_width);
      const double *stop;
      double x;

      stop = src + max (0, i - filt_width);
      x = *stop;
      while (start <= stop) { acc += (*g++) * x; start++; }

      stop =  src + min (width - 1, i + filt_width);
      while (start <  stop) acc += (*g++) * (*start++);

      x = *start;
      stop = src + (i + filt_width);
      while (start <= stop) { acc += (*g++) * x; start++; }

      *dst = acc;
      dst += height;

      assert (g - filt == 2 * filt_width +1);
    }
}

void
Sift::convtransp (double* dst,
            const double* src,
            const double* filt,
            int width, int height, int filt_width)
{
  DEBUG() << "+convtransp" << std::endl;

  const int size = width*height*sizeof (double);
  const int filt_size = (2*filt_width_+1)*sizeof (double);
  double* dev_src = d_malloc<double> (size);
  double* dev_filt = d_malloc<double> (filt_size);
  double* dev_dst = d_malloc<double> (size);

  cudaMemcpy (dev_src, src, size, cudaMemcpyHostToDevice);
  cudaMemcpy (dev_filt, filt_, filt_size, cudaMemcpyHostToDevice);
  _convtransp <<<height, 1>>> (dev_dst, dev_src, dev_filt, width, height, filt_width);
  cudaMemcpy (dst, dev_dst, size, cudaMemcpyDeviceToHost);

  d_free (dev_src);
  d_free (dev_filt);
  d_free (dev_dst);

  DEBUG() << "-convtransp" << std::endl;
}

void
Sift::imsmooth(double* dst,
         double* temp,
         double  const* src,
         int width, int height, double sigma)
{
  DEBUG() << "+imsmooth" << std::endl;

  if (sigma < (double)(1e-5))
    {
      dst = (double*) memcpy (dst,src,width*height*sizeof(double));
      return;
    }

  filt_width_ = (int) ceil (4.0 * sigma);
  DEBUG() << filt_width_ << "/" << filt_res_ << std::endl;

  if (filt_sigma_ != sigma)
    {
      double acc = .0;

      if ((2 * filt_width_ + 1) > filt_res_)
        {
          DEBUG() << "alloc" << std::endl;
          if (!!filt_)
            h_free<double> (filt_);
          filt_ = h_malloc<double>  (sizeof(double) * (2*filt_width_+1));
          filt_res_ = 2 * filt_width_ + 1;
        }
      filt_sigma_ = sigma;

      for (int j = 0; j < 2 * filt_width_ + 1; ++j)
        {
          double  d = (double)(j - filt_width_) / (double)(sigma);
          filt_ [j] = exp (- .5 * d * d);
          acc += filt_ [j];
        }

      for (int j = 0; j < 2 * filt_width_ + 1; ++j)
        filt_ [j] /= acc;
    }

  convtransp (temp, src, filt_,
              width, height, filt_width_);
  convtransp (dst, temp, filt_,
              height, width, filt_width_);
  DEBUG() << "-imsmooth" << std::endl;
}
