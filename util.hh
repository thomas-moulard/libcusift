/**
 * Misc math / memory allocation functions.
 */
#ifndef UTIL_HH_
# define UTIL_HH_
# include <cassert>
# include <cmath>

# include <cstdlib>

/******************************************
 * System                                 *
 ******************************************/

/// Safe host malloc.
template <typename T>
inline T*
h_malloc (size_t s)
{
  return static_cast<T*> (malloc (s));
}


/******************************************
 * Fast math                              *
 ******************************************/
inline double
log2 (int x)
{
  return log (x) / 0.693147180559945;
}

inline int
shift_left(int x, int n)
{
  return (n >= 0) ? (x << n) : (x >> -n);
}

inline double
mod_2pi (double x)
{
  while (x < 0.)
    x += 2 * M_PI ;
  while (x > 2 * M_PI)
    x -= 2 * M_PI ;
  return x;
}

static const int EXPN_SZ = 256;
static const double EXPN_MAX = 25.0;
extern double expn_tab [EXPN_SZ];

void fast_expn_init (double expn_tab[EXPN_SZ]);

inline double
fast_expn (double x, double expn_tab[EXPN_SZ])
{
  double a,b,r;
  int i;
  assert(0 <= x && x <= EXPN_MAX);

  x *= EXPN_SZ / EXPN_MAX;
  i = (int)floor (x);
  r = x - i ;
  a = expn_tab [i];
  b = expn_tab [i + 1];
  return a + r * (b - a);
}

#endif //! UTIL_HH_
