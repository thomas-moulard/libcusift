/**
 * Fast expn implementation.
 */
#include <iostream>
#include "util.hh"

void
fast_expn_init (double expn_tab[EXPN_SZ])
{
  std::cout << "+fast_expn_init" << std::endl;

  for(int k = 0 ; k < EXPN_SZ + 1 ; ++ k)
    expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ));
  std::cout << "-fast_expn_init" << std::endl;
}
