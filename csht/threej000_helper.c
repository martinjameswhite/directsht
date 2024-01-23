#include	<stdlib.h>
#include	<stdio.h>
#include	<math.h>
#include	"omp.h"



#ifdef	ADDJUNK
long	indx(int l1, int l2, int l3) {
int	j1,j2,j3;
long	ii;
  /* Order l1, l2 and l3 such that j1>=j2>=j3. */
  j1 = (l1>l2)?l1:l2;
  j1 = (j1>l3)?j1:l3;
  j3 = (l1<l2)?l1:l2;
  j3 = (j3<l3)?j3:l3;
  j2 = l1 + l2 + l3 - j1 - j3;
  /* And then return the index. */
  return(  (j1*(j1+1L)*(j1+2L))/6 + (j2*(j2+1L))/2 + j3 );
}
#endif






double	calc_3j(int l1, int l2, int l3) {
/* Returns the Wigner 3j symbol for integer ell's and m1=m2=m3=0.  It
   is actually faster to not check for previously computed values to
   avoid needing to synchronize the array among threads. */
int	j1,j2,j3,J;
long	num,den;
double	res,fac;
  J = l1+l2+l3;
  if (J&1>0) return(0);
  /* Order ell1, ell2 and ell3 such that j1>=j2>=j3. */
  j1 = (l1>l2)?l1:l2;
  j1 = (j1>l3)?j1:l3;
  j3 = (l1<l2)?l1:l2;
  j3 = (j3<l3)?j3:l3;
  j2 = l1 + l2 + l3 - j1 - j3;
  /* Check for a terminal case. */
  if ( (j1!=j2)&&(j3==0) ) return(0);
  if ( (j1==j2)&&(j3==0) ) {
    res = 1.0 / sqrt(2*j1+1.0);
    if (j1&1>0) res = -res;
    return(res);
  }
  /* Otherwise we need to recurse. */
  num = (J-2*j2-1L)*(J-2*j3+2L);
  den = (J-2*j2   )*(J-2*j3+1L);
  fac = sqrt((double)num/(double)den);
  res = fac*calc_3j(j1,j2+1,j3-1);
  return(res);
}





int	make_table(int Nl, double store[]) {
/* Makes the three-j table. */
int     ell,j1,j2,j3;
long    ii;
  for (j1=0; j1<Nl; j1++)
    for (j2=0; j2<=j1; j2++)
#pragma omp parallel for private(j3,ii), shared(j1,j2,Nl,store), schedule(static,4)
      for (j3=0; j3<=j2; j3++) {
        ii = (j1*(j1+1L)*(j1+2L))/6 + (j2*(j2+1L))/2 + j3;
        if (j3<j1-j2)
          store[ii] = 0.0;
        else
          store[ii] = calc_3j(j1,j2,j3);
      }
  return(0);
}
