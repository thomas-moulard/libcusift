/**
 * Fast expn implementation.
 */
#include "util.hh"

double expn_tab [EXPN_SZ];

void
fast_expn_init ()
{
  for(int k = 0 ; k < EXPN_SZ + 1 ; ++ k)
    expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ));
}
