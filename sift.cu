/**
 * SIFT implementation.
 */

//FIXME: in extract, return sift points.
//       add cuda, write report, test.

#include "sift.hh"

Sift::Sift (const IplImage& src_, double pt, double te, double nt,
            int O_, int S_, int o_min_)
  : n_keys (0),
    n_keys_res (0),
    keys (0),
    src (src_),
    peak_threshold (pt),
    edge_threshold (te),
    norm_threshold (nt),
    O ((O < 0) ? compute_o_min (o_min_, src_.width, src_.height) : O_),
    S (S_),
    o_min (o_min_),
    s_min (-1),
    s_max (S_ + 1),
    w (src.width),
    h (src.height),
    s (shift_left (w, -o_min) * shift_left (h, -o_min) * sizeof (double)),
    xo (1),
    yo (w),
    so (w * h),
    sigmak_ (pow(2.0, 1.0 / S_)), // 2^(1/S)
    sigman_ (0.5),
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
    im_ (h_malloc<double> (src_.width * src.height * sizeof (double)))
{
  std::cout << "Create SIFT filter with parameter:" << std::endl
            << "* Number of octaves: " << O_ << std::endl
            << "* Min octave: " << o_min_ << std::endl
            << "* S: " << S_ << std::endl
            << "* Peak/Edge/Norm thresholds: " << pt << "/" << te << "/" << nt
            << std::endl;

  // Convert openCV image to double*
  int offset = 0;
  for (int y = 0; y < src.height; ++y)
    for (int x = 0; x < src.width; ++x)
      im_[offset++] = cvGet2D(&src, y, x).val[0];
}

Sift::~Sift ()
{
  free (keys);
  free (im_);
  free (tmp_);
  free (dog_);
  free (gradient_);
  free (octave_);
}

bool
Sift::process ()
{
  std::cout << "+Process" << std::endl;
  if (!O)
    return false;
  int o, s;
  double sa, sb;

  oCur_ = o_min;
  n_keys = 0;
  oW_ = shift_left (src.width,  -oCur_);
  oH_ = shift_left (src.height, -oCur_);
  s = oW_ * oH_;

  double* octave = get_octave (s_min);

  if (o_min < 0)
    {
      /* double once */
      copy_and_upsample_rows (tmp_, im_, w, h);
      copy_and_upsample_rows (octave, tmp_, h, 2 * w);

      /* double more */
      for (o = -1; o > o_min; --o)
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
    memcpy(octave, im_, s);

  // adjust smoothing.
  sa = sigma0_ * pow (sigmak_, s_min);
  sb = sigman_ * pow (2.0, -o_min);

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb);
    imsmooth (octave, tmp_, octave, oW_, oH_, sd);
  }

  // compute octave.
  for(s = s_min + 1; s <= s_max; ++s) {
    double sd = dsigma0_ * pow (sigmak_, s);
    imsmooth (get_octave(s), tmp_,
              get_octave(s - 1), w, h, sd);
  }
  std::cout << "-Process" << std::endl;
  return true;
}

bool
Sift::process_next ()
{
  std::cout << "+Process next" << std::endl;
  if (oCur_ == o_min + O - 1)
    return false;

  int s_best = min (s_min + S, s_max);
  double* pt = get_octave (s_best);
  double* octave = get_octave (s_min);

  copy_and_downsample (octave, pt, w, h, 1);

  oCur_ += 1, n_keys = 0;
  oW_ = shift_left (w,  - oCur_);
  oH_ = shift_left (h, - oCur_);

  double sa = sigma0_ * powf (sigmak_, s_min);
  double sb = sigma0_ * powf (sigmak_, s_best - S);

  if (sa > sb)
    {
      double sd = sqrt (sa*sa - sb*sb);
      imsmooth (octave, tmp_, octave, oW_, oH_, sd);
    }
  std::cout << "-Process next" << std::endl;
  return true;
}

#define SINGLE_EPSILON 1.19209290E-07F
double
Sift::normalize_histogram (double *begin, double *end)
{
  double norm = 0.0;
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

  const double magnif      = 3.0;
  const int    NBO         = 8;
  const int    NBP         = 4;
  double       xper        = pow (2.0, oCur_);
  const int    xo          = 2;         /* x-stride */
  const int    yo          = 2 * w;     /* y-stride */
  const int    so          = 2 * w * h; /* s-stride */
  double       x           = k->x     / xper;
  double       y           = k->y     / xper;
  double       sigma       = k->sigma / xper;

  int          xi          = (int) (x + 0.5);
  int          yi          = (int) (y + 0.5);
  int          si          = k->is;

  const double st0         = sin (angle);
  const double ct0         = cos (angle);
  const double SBP         = magnif * sigma;
  const int W           = (int)floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5);

  const int binto = 1;          /* bin theta-stride */
  const int binyo = NBO * NBP;  /* bin y-stride */
  const int binxo = NBO;        /* bin x-stride */

  /* check bounds */
  if(k->o  != oCur_ || xi <  0 || xi >= w || yi <  0 ||
     yi >= h - 1 || si < s_min + 1 || si > s_max - 2)
    return;

  /* synchronize gradient buffer */
  update_gradient ();

  /* clear descriptor */
  memset (descr, 0, sizeof(double) * NBO*NBP*NBP);

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  const double* pt  = gradient_ + xi*xo + yi*yo + (si - s_min - 1)*so;
  double* dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo;

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(int dyi =  max (- W, 1 - yi); dyi <= min (+ W, h - yi - 2); ++dyi)
    for(int dxi =  max (- W, 1 - xi); dxi <= min (+ W, w - xi - 2); ++ dxi)
      {
        /* retrieve */
        double mod   = *( pt + dxi*xo + dyi*yo + 0 );
        double angle_ = *( pt + dxi*xo + dyi*yo + 1 );
        double theta = mod_2pi (angle_ - angle);

        /* fractional displacement */
        double dx = xi + dxi - x;
        double dy = yi + dyi - y;

        /* get the displacement normalized w.r.t. the keypoint
           orientation and extension */
        double nx = ( ct0 * dx + st0 * dy) / SBP;
        double ny = (-st0 * dx + ct0 * dy) / SBP;
        double nt = NBO * theta / (2 * M_PI);

        /* Get the Gaussian weight of the sample. The Gaussian window
         * has a standard deviation equal to NBP/2. Note that dx and dy
         * are in the normalized frame, so that -NBP/2 <= dx <=
         * NBP/2. */
        double const wsigma = NBP/2;
        double win = fast_expn
          ((nx*nx + ny*ny)/(2.0 * wsigma * wsigma));

        /* The sample will be distributed in 8 adjacent bins.
           We start from the ``lower-left'' bin. */
        int binx = (int)floor (nx - 0.5);
        int biny = (int)floor (ny - 0.5);
        int bint = (int)floor (nt);
        double rbinx = nx - (binx + 0.5);
        double rbiny = ny - (biny + 0.5);
        double rbint = nt - bint;

        /* Distribute the current sample into the 8 adjacent bins*/
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

  /* Normalize the histogram to L2 unit length. */
  double norm = normalize_histogram (descr, descr + NBO*NBP*NBP);

  /* Set the descriptor to zero if it is lower than our norm_threshold */
  if(norm_threshold && norm <  norm_threshold)
    for (int bin = 0; bin < NBO*NBP*NBP; ++ bin)
      descr [bin] = 0;
  else
    {
      /* Truncate at 0.2. */
      for(int bin = 0; bin < NBO*NBP*NBP; ++ bin)
        if (descr [bin] > 0.2) descr [bin] = 0.2;

      /* Normalize again. */
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

      for (int i=0; i<n_keys; ++i)
        {
          double angles[4];
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

void
Sift::compute_dog ()
{
  double* pt = dog_;
  for (int s = s_min; s <= s_max - 1; ++s)
    {
      double* src_a = get_octave (s);
      double* src_b = get_octave (s + 1);
      double* end_a = src_a + oW_ * oH_;
      while (src_a != end_a)
        *pt++ = *src_b++ - *src_a++;
    }
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
  std::cout << "+Detect maxima" << std::endl;
  double* pt  = dog_ + xo + yo + so;

  for (int s = s_min + 1; s <= s_max - 2; ++s)
    {
      for(int y = 1; y < h - 1; ++y)
        {
          for(int x = 1; x < w - 1; ++x)
            {
              double v = *pt;
              if (CHECK_NEIGHBORS(>,+) ||
                  CHECK_NEIGHBORS(<,-) )
                {
                  /* make room for more keypoints */
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
                  k->ix = x;
                  k->iy = y;
                  k->is = s;
                }
              pt += 1;
            }
          pt += 2;
        }
      pt += 2 * yo;
    }
  std::cout << "-Detect maxima" << std::endl;
}

#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))
#define Aat(i,j)     (A[(i)+(j)*3])
void
Sift::refine_maxima ()
{
  std::cout << "+Refine maxima" << std::endl;
  SiftKeypoint* k = keys;
  int i = 0, j = 0, jj = 0, ii = 0;

  double maxa = 0;
  double maxabsa = 0;
  int maxi  = -1;
  double tmp = 0.;
  double* pt = 0;
  double xper  = pow (2.0, oCur_);

  for (i = 0; i < n_keys; ++i)
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
          x += dx;
          y += dy;

          pt = dog_ + xo * x + yo * y + so * (s - s_min);

          Dx = 0.5 * (at(+1,0,0) - at(-1,0,0));
          Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
          Ds = 0.5 * (at(0,0,+1) - at(0,0,-1));

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

          /* Gauss elimination */
          for(j = 0; j < 3; ++j)
            {
              /* look for the maximally stable pivot */
              for (int i = j; i < 3; ++i)
                {
                  double a = Aat (i,j), absa = abs (a);
                  if (absa > maxabsa)
                    maxa = a, maxabsa = absa, maxi = i;
                }
            }

          /* if singular give up */
          if (maxabsa < 1e-10f)
            {
              b[0] = 0, b[1] = 0, b[2] = 0;
              break;
            }

          i = maxi;

          /* swap j-th row with i-th row and normalize j-th row */
          for(jj = j; jj < 3; ++jj)
            tmp = Aat(i,jj); Aat(i,jj) = Aat(j,jj); Aat(j,jj) = tmp,
                                                      Aat(j,jj) /= maxa;
          tmp = b[j]; b[j] = b[i]; b[i] = tmp;
          b[j] /= maxa;

          /* elimination */
          for (ii = j+1; ii < 3; ++ii)
            {
              double x = Aat(ii,j);
              for (jj = j; jj < 3; ++jj)
                Aat(ii,jj) -= x * Aat(j,jj);
              b[ii] -= x * b[j];
            }

          /* backward substitution */
          for (i = 2; i > 0; --i)
            {
              double x = b[i];
              for (ii = i-1; ii >= 0; --ii)
                b[ii] -= x * Aat(ii,i);
            }

          /* ........................................................... */
          /* If the translation of the keypoint is big, move the keypoint
           * and re-iterate the computation. Otherwise we are all set.
           */

          dx= ((b[0] >  0.6 && x < w - 2) ?  1 : 0)
            + ((b[0] < -0.6 && x > 1    ) ? -1 : 0);

          dy= ((b[1] >  0.6 && y < h - 2) ?  1 : 0)
            + ((b[1] < -0.6 && y > 1    ) ? -1 : 0);

          if (dx == 0 && dy == 0) break;
        }

      /* check threshold and other conditions */
      {
        double val   = at(0,0,0)
          + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
        double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy);
        double xn = x + b[0];
        double yn = y + b[1];
        double sn = s + b[2];

        bool good =
          abs (val)  > peak_threshold           &&
          score           < (edge_threshold+1)*(edge_threshold+1)
          /edge_threshold    &&
          score           >= 0                  &&
          abs (b[0]) <  1.5                     &&
          abs (b[1]) <  1.5                     &&
          abs (b[2]) <  1.5                     &&
          xn              >= 0                  &&
          xn              <= w - 1              &&
          yn              >= 0                  &&
          yn              <= h - 1              &&
          sn              >= s_min              &&
          sn              <= s_max;

        if (good) {
          k->o = oCur_;
          k->ix = x, k->iy = y, k->is = s;
          k->s = sn;
          k->x = xn * xper, k->y = yn * xper;
          k->sigma = sigma0_ * pow (2.0, sn/S) * xper;
          ++k;
        }

      } /* done checking */
    } /* next keypoint to refine */

  /* update keypoint count */
  n_keys = k - keys;
  std::cout << "-Refine maxima" << std::endl;
}
#undef at

#define SAVE_BACK                                       \
  *gradient_++ = sqrt (gx*gx + gy*gy);                  \
  *gradient_++ = mod_2pi(atan2 (gy, gx) + 2*M_PI);      \
  ++src


void
Sift::update_gradient ()
{
  if (oGrad_ == oCur_)
    return;

  for (int s  = s_min + 1;
       s <= s_max - 2; ++ s)
    {
      double* src;
      double* end;
      double gx, gy;
      gradient_ = gradient_ + 2 * so * (s - s_min -1);
      src  = get_octave (s);

      /* first first row */
      gx = src[+xo] - src[0];
      gy = src[+yo] - src[0];
      SAVE_BACK;

      /* middle first row */
      end = (src - 1) + w - 1;
      while (src < end)
        {
          gx = 0.5 * (src[+xo] - src[-xo]);
          gy =        src[+yo] - src[0];
          SAVE_BACK;
        }

      /* first first row */
      gx = src[0]   - src[-xo];
      gy = src[+yo] - src[0];
      SAVE_BACK;

      for (int y = 1; y < h -1; ++y)
        {
          /* first middle row */
          gx =        src[+xo] - src[0];
          gy = 0.5 * (src[+yo] - src[-yo]);
          SAVE_BACK;

          /* middle middle row */
          end = (src - 1) + w - 1;
          while (src < end)
            {
              gx = 0.5 * (src[+xo] - src[-xo]);
              gy = 0.5 * (src[+yo] - src[-yo]);
              SAVE_BACK;
            }

          /* last middle row */
          gx =        src[0]   - src[-xo];
          gy = 0.5 * (src[+yo] - src[-yo]);
          SAVE_BACK;
        }

      /* first last row */
      gx = src[+xo] - src[0];
      gy = src[  0] - src[-yo];
      SAVE_BACK;

      /* middle last row */
      end = (src - 1) + w - 1;
      while (src < end)
        {
          gx = 0.5 * (src[+xo] - src[-xo]);
          gy =        src[0]   - src[-yo];
          SAVE_BACK;
        }

      /* last last row */
      gx = src[0]   - src[-xo];
      gy = src[0]   - src[-yo];
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

  const int xo = 2;         /* x-stride */
  const int yo = 2 * w;     /* y-stride */
  const int so = 2 * w * h; /* s-stride */
  double x = k->x/xper, y = k->y/xper;
  double sigma = k->sigma/xper;

  int xi = (int) (x + 0.5),
    yi = (int) (y + 0.5), si = k->is;

  const double sigmaw = winf * sigma;
  int W = (int)max (floor (3.0 * sigmaw), 1.);
  int nangles = 0;
  enum { nbins = 36 };

  double hist [nbins], maxh;

  /* skip if the keypoint octave is not current */
  if(k->o != oCur_)
    return 0;

  /* skip the keypoint if it is out of bounds */
  if(xi < 0 || xi > w - 1 || yi < 0 || yi > h - 1 ||
     si < s_min + 1 || si > s_max - 2 )
    return 0;

  /* make gradient up to date */
  update_gradient ();

  /* clear histogram */
  memset (hist, 0, sizeof(double) * nbins);

  /* compute orientation histogram */
  double* pt =  gradient_ + xo*xi + yo*yi + so*(si - s_min - 1);

  for (int ys  =  max (- W, - yi);
       ys <=  min (+ W, h - 1 - yi); ++ys)
    for (int xs  = max (- W,       - xi);
         xs <= min (+ W, w - 1 - xi); ++xs)
      {
        double dx = (double)(xi + xs) - x;
        double dy = (double)(yi + ys) - y;
        double r2 = dx*dx + dy*dy;
        double wgt, mod, ang, fbin;

        /* limit to a circular window */
        if (r2 >= W*W + 0.6)
          continue;

        wgt  = fast_expn (r2 / (2*sigmaw*sigmaw));
        mod  = *(pt + xs*xo + ys*yo    );
        ang  = *(pt + xs*xo + ys*yo + 1);
        fbin = nbins * ang / (2 * M_PI);

        int    bin  = (int)floor (fbin - 0.5);
        double rbin = fbin - bin - 0.5;
        hist [(bin + nbins) % nbins] += (1 - rbin) * mod * wgt;
        hist [(bin + 1    ) % nbins] += (rbin) * mod * wgt;
      }

  /* smooth histogram */
  for (int iter = 0; iter < 6; iter++)
    {
      double prev  = hist [nbins - 1];
      double first = hist [0];
      int i;
      for (i = 0; i < nbins - 1; i++)
        {
          double newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
          prev = hist[i];
          hist[i] = newh;
        }
      hist[i] = (prev + hist[i] + first) / 3.0;
    }

  /* find the histogram maximum */
  maxh = 0;
  for (int i = 0; i < nbins; ++i)
    maxh = max (maxh, hist [i]);

  /* find peaks within 80% from max */
  int n_angles = 0;
  for (int i = 0; i < nbins; ++i)
    {
      double h0 = hist [i];
      double hm = hist [(i - 1 + nbins) % nbins];
      double hp = hist [(i + 1 + nbins) % nbins];

      /* is this a peak? */
      if (h0 > 0.8*maxh && h0 > hm && h0 > hp)
        {
          /* quadratic interpolation */
          double di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0);
          double th = 2 * M_PI * (i + di + 0.5) / nbins;
          angles [ nangles++ ] = th;
          if( nangles == 4 )
            return n_angles;
        }
    }
  return n_angles;
}
#undef at

void
Sift::copy_and_upsample_rows (double* dst,
                             double const* src,
                             int width, int height)
{
  std::cout << "+copy_and_upsample_rows" << std::endl;
  int x, y;
  double a, b;

  for (y = 0; y < height; ++y)
    {
      b = a = *src++;
      for (x = 0; x < width - 1; ++x)
        {
          b = *src++;
          *dst = a; dst += height;
          *dst = 0.5 * (a + b); dst += height;
          a = b;
        }
      *dst = b; dst += height;
      *dst = b; dst += height;
      dst += 1 - width * 2 * height;
    }
  std::cout << "-copy_and_upsample_rows" << std::endl;
}

void
Sift::copy_and_downsample (double* dst,  double const* src,
                           int width, int height, int d)
{
  std::cout << "+copy_and_downsample" << std::endl;
  int x, y;

  d = 1 << d; /* d = 2^d */
  for (y = 0; y < height; y+=d)
    {
      double const* srcrowp = src + y * width;
      for(x = 0; x < width - (d-1); x+=d)
        {
          *dst++ = *srcrowp;
          srcrowp += d;
        }
    }
  std::cout << "-copy_and_downsample" << std::endl;
}

void
Sift::convtransp (double* dst,
            double const* src,
            double const* filt,
            int width, int height, int filt_width)
{
  std::cout << "+convtransp " << width << "/" << height << "/" << filt_width << std::endl;
  int i, j;

  for(j = 0; j < height; ++j) {
    for(i = 0; i < width; ++i) {
      double        acc   = 0.0;
      double const *g     = filt;
      double const *start = src + (i - filt_width);
      double const *stop;
      double        x;

      /* beginning */
      //std::cout << "+b" << std::endl;
      stop = src + max (0, i - filt_width);
      x    = *stop;
      while (start <= stop) { acc += (*g++) * x; start++; }
      //std::cout << "-b" << std::endl;

      /* middle */
      //std::cout << "+m" << std::endl;
      stop =  src + min (width - 1, i + filt_width);
      while (start <  stop) acc += (*g++) * (*start++);
      //std::cout << "-m" << std::endl;

      /* end */
      //std::cout << "+e" << std::endl;
      x  = *start;
      stop = src + (i + filt_width);
      while (start <= stop) { acc += (*g++) * x; start++; }
      //std::cout << "-e" << std::endl;

      /* save */
      *dst = acc; //FIXME??
      dst += height;

      assert (g - filt == 2 * filt_width +1);
    }
    /* next column */
    src += width;
    dst -= width*height - 1;
  }
  std::cout << "-convtransp" << std::endl;
}

void
Sift::imsmooth(double* dst,
         double* temp,
         double  const* src,
         int width, int height, double sigma)
{
  std::cout << "+imsmooth " << width << "/" << height << "/" << sigma << std::endl;
  static double   *filt       = 0;
  static int    filt_width = -1;
  static double filt_static_sigma = -1.0;
  enum          { filt_static_res = 1024 };
  static double    filt_static [2 * filt_static_res + 1];

  int j;

  if (sigma < (double)(1e-5))
    {
      dst = (double*) memcpy (dst,src,width*height*sizeof(double));
      return;
    }

  /* window width */
  filt_width = (int) ceil (4.0 * sigma);

  /* setup filter only if not available from previous iteration*/
  if (filt_static_sigma != sigma)
    {
      double acc = 0.0;

      if (filt_width <= filt_static_res)
        {
          /* use static buffer */
          filt = filt_static;
          filt_static_sigma = sigma;
        } else {
        /* dynamically allocate a larger buffer */
        filt = h_malloc<double>  (sizeof(double) * (2*filt_width+1));
      }

      for (j = 0; j < 2 * filt_width + 1; ++j) {
        double  d = (double)(j - filt_width) / (double)(sigma);
        filt [j] = exp (- 0.5 * d * d);
        acc += filt [j];
      }

      /* normalize */
      for (j = 0; j < 2 * filt_width + 1; ++j)
        filt [j] /= acc;
    }

  /* convolve */
  convtransp (temp, src, filt,
              width, height, filt_width);
  convtransp (dst, temp, filt,
              height, width, filt_width);
  /* free buffer? */
  if (filt_static_sigma != sigma)
    free (filt);
  std::cout << "-imsmooth" << std::endl;
}
