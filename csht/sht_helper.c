#include	<stdlib.h>
#include	<stdio.h>
#include	<math.h>
#include	"omp.h"



long	indx(int ell, int m, int Nl) {
long	ii;
  ii = 2*Nl-1-m;
  ii = (m*ii)/2 + ell;
  return(ii);
}




int	make_table(int Nl, int Nx, double xmax, double Yv[], double Yd[]) {
/* Makes the function value and derivative tables: Yv, Yd. */
int     ell,m,ix;
long    ii,i0,i1,i2;
double	xx,omx2,dx,fact1,fact2;
  dx = xmax/(Nx-1.0);
  /* First do Legendre polynomials. */
#pragma omp parallel for private(ix), shared(Nx,dx,Yv)
  for (ix=0; ix<Nx; ix++) {
    Yv[Nx*0+ix] = 1.0;
    Yv[Nx*1+ix] = ix*dx;
  }
  for (ell=2; ell<Nl; ell++) {
    i0 = Nx*indx(ell-0,0,Nl);
    i1 = Nx*indx(ell-1,0,Nl);
    i2 = Nx*indx(ell-2,0,Nl);
#pragma omp parallel for private(ix,xx), shared(Nl,Nx,dx,ell,i0,i1,i2,Yv)
    for (ix=0; ix<Nx; ix++) {
      xx        = ix * dx;
      Yv[i0+ix] = (2-1./ell)*xx*Yv[i1+ix] - (1.0-1./ell)*Yv[i2+ix];
    }
  }
  /* Now fill in m=ell function values. */
  for (m=1; m<Nl; m++) {
    i0 = Nx*indx(m-0,m-0,Nl);
    i1 = Nx*indx(m-1,m-1,Nl);
#pragma omp parallel for private(ix,xx), shared(Nl,Nx,dx,m,i0,i1,Yv)
    for (ix=0; ix<Nx; ix++) {
      xx        = ix * dx;
      Yv[i0+ix] = -sqrt( (1.0-0.5/m)*(1.0-xx*xx) )*Yv[i1+ix];
    }
  }
  /* and the m=ell-1 function values. */
  for (m=1; m<Nl-1; m++) {
    i0 = Nx*indx(m+0,m,Nl);
    i1 = Nx*indx(m+1,m,Nl);
#pragma omp parallel for private(ix,xx), shared(Nl,Nx,dx,m,i0,i1,Yv)
    for (ix=0; ix<Nx; ix++) {
      xx        = ix * dx;
      Yv[i1+ix] = sqrt(2*m+1.)*xx*Yv[i0+ix];
    }
  }
  /* Finally fill in the other m values. */
  for (m=0; m<Nl-1; m++)
    for (ell=m+2; ell<Nl; ell++) {
      i0    = Nx*indx(ell-0,m,Nl);
      i1    = Nx*indx(ell-1,m,Nl);
      i2    = Nx*indx(ell-2,m,Nl);
      fact1 = sqrt( (double)(ell-m)/(double)(ell+m) );
      fact2 = sqrt( (ell-m-1.)/(ell+m-1.) );
#pragma omp parallel for private(ix,xx), shared(Nl,Nx,dx,ell,m,i0,i1,i2,fact1,fact2,Yv), schedule(static)
      for (ix=0; ix<Nx; ix++) {
        xx        = ix * dx;
        Yv[i0+ix] = (2*ell-1)*xx*Yv[i1+ix] - (ell+m-1)*Yv[i2+ix]*fact2;
        Yv[i0+ix]*= fact1/(ell-m);
      }
    }
  /* Then we do the derivatives -- again do m=0 separately. */
  for (ix=0; ix<Nx; ix++) {
    Yd[Nx*0+ix] = 0.0;
    Yd[Nx*1+ix] = 1.0;
  }
#pragma omp parallel for private(ell,i0,i1,ix,xx,omx2), shared(Nl,Nx,dx,Yv,Yd)
  for (ell=2; ell<Nl; ell++) {
    i0 = Nx*indx(ell-0,0,Nl);
    i1 = Nx*indx(ell-1,0,Nl);
    for (ix=0; ix<Nx; ix++) {
      xx        = ix * dx;
      omx2      = 1.0-xx*xx;
      Yd[i0+ix] = ell*(Yv[i1+ix]-xx*Yv[i0+ix])/omx2;
    }
  }
  /* Then the higher m's. */
#pragma omp parallel for private(ell,m,i0,i1,fact1,ix,xx,omx2), shared(Nl,Nx,dx,Yv,Yd)
  for (ell=0; ell<Nl; ell++)
    for (m=1; m<=ell; m++) {
      i0    = Nx*indx(ell-0,m,Nl);
      i1    = Nx*indx(ell-1,m,Nl);
      fact1 = sqrt( (double)(ell-m)/(double)(ell+m) );
      for (ix=0; ix<Nx; ix++) {
        xx        = ix * dx;
        omx2      = 1.0-xx*xx;
        Yd[i0+ix] = ((ell+m)*fact1*Yv[i1+ix]-ell*xx*Yv[i0+ix])/omx2;
      }
    }
  /* Normalize by Sqrt[ (2ell+1)/4Pi ] */
#pragma omp parallel for private(ell,m,fact1,ii,ix), shared(Nl,Nx,Yv,Yd), schedule(static)
  for (ell=0; ell<Nl; ell++) {
    fact1 = sqrt( (2*ell+1.0)/4./M_PI );
    for (m=0; m<=ell; m++) {
      ii = Nx*indx(ell,m,Nl);
      for (ix=0; ix<Nx; ix++) {
        Yv[ii+ix] *= fact1;
        Yd[ii+ix] *= fact1;
      }
    }
  }
  return(0);
}



#ifdef	SMALLMEM



int	do_transform(int Nl, int Nx, double xmax, double Yv[], double Yd[],
                     int Np, double cost[], double phi[], double wt[],
                     double carr[], double sarr[]) {
/* Does the direct SHT, filling in the cosine and sine arrays.  This
   version keeps memory usage to a minimum at the expense of redoing
   the interpolation in x for each (ell,m), which is slow. */
int     ell,m,ix;
long    offset,ii,i0,i1;
double	xx,ax,dx,sc,ss,yv;
double	tt,t1,t2,s0,s1,s2,s3;
  /* Zero the carr and sarr -- not really necessary. */
#pragma omp parallel for private(ii) shared(Nl,carr,sarr)
  for (ii=0; ii<(Nl*(Nl+1))/2; ii++) {
    carr[ii]=sarr[ii]=0.0;
  }
  dx = xmax/(Nx-1.0);
  for (ell=0; ell<Nl; ell++)
    for (m=0; m<=ell; m++) {
      offset  = Nx*indx(ell,m,Nl);
      ss = sc = 0.0; /* Assumulate the sine and cosine sums here.*/
#pragma omp parallel for private(ii,ix,i0,i1,xx,ax,tt,t1,t2,s0,s1,s2,s3,yv), shared(Nl,Np,ell,m,dx,offset,Yv,Yd,cost,phi,wt), reduction(+:sc,ss), schedule(static)
      for (ii=0; ii<Np; ii++) {
        /* Use Hermite spline to get Ylm(x,0). */
        xx = cost[ii];
        ax = fabs(xx);
        ix = ax/dx;
        tt = ax/dx-ix;
        t1 = (tt-1.0)*(tt-1.0);
        t2 = tt*tt;
        s0 = (1+2*tt)*t1;
        s1 = tt*t1;
        s2 = t2*(3-2*tt);
        s3 = t2*(tt-1.0);
        i0 = offset+ix;
        i1 = i0+1;
        yv = Yv[i0]*s0+Yd[i0]*s1*dx+Yv[i1]*s2+Yd[i1]*s3*dx;
        /* Flip sign if x<0 and ell-m is odd. */
        if (xx<0 && (ell-m)%2==1) yv = -yv;
        /* Multiply through by sin(m.phi) and cos(m.phi) */
        sc+= wt[ii] * yv * cos(m*phi[ii]);
        ss+= wt[ii] * yv * sin(m*phi[ii]);
      }
      carr[indx(ell,m,Nl)] = sc;
      sarr[indx(ell,m,Nl)] = ss;
    }
  return(0);
}


#else
  

int	do_transform(int Nl, int Nx, double xmax, double Yv[], double Yd[],
                 int Np, double cost[], double phi[], double wt[],
                 double carr[], double sarr[]) {
/* Does the direct SHT, filling in the cosine and sine arrays.  This
   version is faster than the above, but uses more memory.  The objects
   must also be sorted in order of increasing cost, and it is assumed
   the phase factor for cost<0 will be applied externally. */
int     ell,m,ix;
int     ithread,nthread;
long    ii,jj,i0,i1,jmin,jmax,Nlm;
double  xx,ax,dx;
double  tt,t1,t2,s0,s1,s2,s3;
double  *cj,*sj,*csum,*ssum;
  /* Make storage for the intermediate "cj" arrays. */
  nthread = omp_get_max_threads();
  ii = 4L*Nl*Nx;
  cj = malloc(ii*nthread*sizeof(double));
  if (cj==NULL) {perror("malloc");return(1);}
  sj = malloc(ii*nthread*sizeof(double));
  if (sj==NULL) {perror("malloc");return(1);}
  /* and zero them. */
  for (ii=0; ii<4*Nl*Nx*nthread; ii++) cj[ii]=sj[ii]=0;
  dx = xmax/(Nx-1.0);
  for (jmin=ix=0; ix<Nx; ix++) {
    xx = ix * dx;
    while (jmin<Np && fabs(cost[jmin])< xx  ) jmin++;
    jmax = jmin;
    while (jmax<Np && fabs(cost[jmax])<xx+dx) jmax++;
#pragma omp parallel for private(ii,i0,i1,ithread,m,ax,tt,t1,t2,s0,s1,s2,s3), shared(Nx,Nl,jmin,jmax,dx,cost,phi,wt,cj,sj), schedule(static)
    for (ii=jmin; ii<jmax; ii++) {
      ithread = omp_get_thread_num();
      ax = fabs(cost[ii]);
      i0 = ax/dx;
      tt = ax/dx-i0;
      t1 = (tt-1.0)*(tt-1.0);
      t2 = tt*tt;
      s0 = (1+2*tt)*t1;
      s1 = tt*t1*dx;
      s2 = t2*(3-2*tt);
      s3 = t2*(tt-1.0)*dx;
      i1 = ithread*(4*Nl*Nx)+0*Nl*Nx+Nl*ix;
      for (m=0; m<Nl; m++) {
        /* We could potentially use trig identities for
           cos(m phi) and sin(m phi) to save time. */
        cj[i1+m] += wt[ii]*s0*cos(m*phi[ii]);
        sj[i1+m] += wt[ii]*s0*sin(m*phi[ii]);
      }
      i1 = ithread*(4*Nl*Nx)+1*Nl*Nx+Nl*ix;
      for (m=0; m<Nl; m++) {
        cj[i1+m] += wt[ii]*s1*cos(m*phi[ii]);
        sj[i1+m] += wt[ii]*s1*sin(m*phi[ii]);
      }
      i1 = ithread*(4*Nl*Nx)+2*Nl*Nx+Nl*ix;
      for (m=0; m<Nl; m++) {
        cj[i1+m] += wt[ii]*s2*cos(m*phi[ii]);
        sj[i1+m] += wt[ii]*s2*sin(m*phi[ii]);
      }
      i1 = ithread*(4*Nl*Nx)+3*Nl*Nx+Nl*ix;
      for (m=0; m<Nl; m++) {
        cj[i1+m] += wt[ii]*s3*cos(m*phi[ii]);
        sj[i1+m] += wt[ii]*s3*sin(m*phi[ii]);
      }
    }
    jmin = jmax;
  }
  for (ithread=1; ithread<nthread; ithread++) {
    for (ii=0; ii<4*Nl*Nx; ii++) {
      cj[ii] += cj[ithread*(4*Nl*Nx)+ii];
      sj[ii] += sj[ithread*(4*Nl*Nx)+ii];
    }
  }
  /* Now do the sums over x-bins. */
  Nlm  = (Nl*(Nl+1))/2;
  csum = malloc(Nlm*nthread*sizeof(double));
  if (csum==NULL) {perror("malloc");return(1);}
  ssum = malloc(Nlm*nthread*sizeof(double));
  if (ssum==NULL) {perror("malloc");return(1);}
#pragma omp parallel for private(jj) shared(Nlm,nthread,csum,ssum)
  for (jj=0; jj<Nlm*nthread; jj++) {csum[jj]=ssum[jj]=0.0;}
  for (ell=0; ell<Nl; ell++)
    for (m=0; m<=ell; m++) {
      ii = indx(ell,m,Nl);
#pragma omp parallel for private(jj,i0,i1,s0,s1,s2,s3) shared(Nl,Nx,Nlm,nthread,csum,ssum)
      for (i0=0; i0<Nx-1; i0++) {
        ithread = omp_get_thread_num();
        jj = ithread*Nlm + ii;
        i1 = i0+1;
        s0 = cj[0*Nl*Nx+Nl*i0+m];
        s1 = cj[1*Nl*Nx+Nl*i0+m];
        s2 = cj[2*Nl*Nx+Nl*i0+m];
        s3 = cj[3*Nl*Nx+Nl*i0+m];
        csum[jj] += Yv[Nx*ii+i0]*s0+Yd[Nx*ii+i0]*s1+Yv[Nx*ii+i1]*s2+Yd[Nx*ii+i1]*s3;                 
        s0 = sj[0*Nl*Nx+Nl*i0+m];
        s1 = sj[1*Nl*Nx+Nl*i0+m];
        s2 = sj[2*Nl*Nx+Nl*i0+m];
        s3 = sj[3*Nl*Nx+Nl*i0+m];
        ssum[jj] += Yv[Nx*ii+i0]*s0+Yd[Nx*ii+i0]*s1+Yv[Nx*ii+i1]*s2+Yd[Nx*ii+i1]*s3;
      }
    }
  free(sj);free(cj);
#pragma omp parallel for private(ii) shared(ix,carr,sarr) schedule(static)
  for (ii=0; ii<Nlm; ii++) {carr[ii]=sarr[ii]=0.0;}
  for (ithread=0; ithread<nthread; ithread++) {
    for (ii=0; ii<Nlm; ii++) {
      carr[ii] += csum[ithread*Nlm+ii];
      sarr[ii] += ssum[ithread*Nlm+ii];
    }
  }
  free(ssum);free(csum);
  return(0);
}

#endif
