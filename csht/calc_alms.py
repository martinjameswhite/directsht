import numpy  as np
import healpy as hp

#
# A thin wrapper to the SHT class that rotates polar points
# to the equator, computes the SHT in those coordinates and
# then rotates back.  Avoids any interpolation issues near
# |cos(theta)|~1.
# Uses healpy for the rotations.
#

def calc_alms(t,p,w,sht):
    """Handles the polar/equatorial split and rotations."""
    alms = np.zeros(sht.Nlm,dtype='complex')
    xval = np.abs(np.cos(t))
    pol  = np.nonzero(xval>=sht.xmax)[0]
    equ  = np.nonzero(xval< sht.xmax)[0]
    print("** Pol/Equ is {:d}/{:d}.".format(len(pol),len(equ)),flush=True)
    if len(pol)>0:
        rotn = (0,90,0)
        yrot = hp.rotator.Rotator(rot=rotn,eulertype='ZYZ',deg=True)
        tp,pp= yrot(t[pol],p[pol])
        rlms = sht(tp,pp,w[pol])
        yrot = hp.rotator.Rotator(rot=rotn,eulertype='ZYZ',deg=True,inv=True)
        alms+= yrot.rotate_alm(rlms)
    if len(equ)>0:
        alms+= sht(t[equ],p[equ],w[equ])
    return(alms)
    #
