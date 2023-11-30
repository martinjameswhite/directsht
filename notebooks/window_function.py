#!/usr/bin/env python
#
# Generate the window function by direct SHT of
# random point sets, and see how it depends upon
# the number of random points.
#
import numpy  as np
import healpy as hp
import sys
#
import matplotlib.pyplot as plt
#
sys.path.append('../sht')
from sht import DirectSHT
#

# Get a random number generator.
rng = np.random.default_rng()


def make_catalog(nrand):
    """Generate a random catalog."""
    trand = np.arccos( rng.uniform(low=-0.5,high=0.5,size=nrand) )
    prand = rng.uniform(low=np.pi/2,high=3*np.pi/2,size=nrand)
    wrand = np.ones_like(trand) / float(nrand)
    return( (trand,prand,wrand) )
    #


def make_cl(nrand,Nl=256):
    """Compute the random/window pseudo-power spectrum."""
    # Set up an sht instance.  We will set Nx to be quite large, so
    # that we are # confident in our interpolation.
    Nx  = 4*Nl+1
    xmax= 9.0/16.0
    sht = DirectSHT(Nl,Nx,xmax)
    # Generate the random catalog.
    trand,prand,wrand = make_catalog(nrand)
    # Now do the SHT and compute wl.
    hran = sht(trand,prand,wrand)
    wl   = hp.alm2cl(hran)
    ell  = np.arange(wl.size)
    return( (ell,wl) )
    #



def make_plot():
    """Does the work of making the figure."""
    fig,ax = plt.subplots(2,1,sharex=True,figsize=(8,5),\
               gridspec_kw={'height_ratios':[2,1]})
    # Set our problem size.
    Nl = 256
    # First compute a "truth" using many randoms.
    nrand  = 4 * 1024 * 1024
    ell,wl = make_cl(nrand,Nl)
    truth  = wl[2:] - 1.0/float(nrand)/(4*np.pi)
    # and iterate over the number of randoms.
    icol = 0
    for irand in [16,18,20]:
        nrand  = 2**irand
        lbl    = r'$\log_2 '+'N={:2d}$'.format(irand)
        ell,wl = make_cl(nrand,Nl)
        ell,wl = ell[2:],wl[2:]
        #
        sn = 1.0/float(nrand) / (4*np.pi)
        #
        ax[0].plot(ell,wl,'s-',color='C'+str(icol),label=lbl)
        ax[0].axhline(sn,ls=':',color='C'+str(icol))
        #
        ax[1].plot(ell,(wl-sn)/truth,'s-',color='C'+str(icol))
        #
        icol = (icol+1)%10
    ax[0].legend()
    ax[0].set_xlim(ell[0],ell[-1])
    ax[1].set_ylim(0.5,1.5)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('linear')
    ax[1].set_xlabel(r'$\ell$')
    ax[0].set_ylabel(r'$W_\ell$')
    ax[1].set_ylabel(r'Ratio')
    #
    plt.tight_layout()
    plt.savefig('window_function.pdf')
    #



if __name__=="__main__":
    make_plot()
    #
