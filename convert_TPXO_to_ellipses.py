# coding: utf-8
import numpy as np
from netCDF4 import Dataset
import scipy.interpolate as interp

def ellipse(u_a,u_p,v_a,v_p):
    """
    CV 2016/10/21 : compute ellipses parameters given the amplitude and phase of u,v
                adapted from matlab's function by Zhigang Xu [see also croco tools]
                _a : amplitude [degrees, not radians]  
                _p : phase     [degrees, not radians]  
    """
    # convert phase from angle to radians 
    u_p = u_p*np.pi/180
    v_p = v_p*np.pi/180
    # complex amplitude for u,v
    u = u_a*np.exp(-1j*u_p)
    v = v_a*np.exp(-1j*v_p)
    # complex radius of anticlockwise and clockwise circles
    wp = (u+1j*v)/2          # anticlockwise circle
    wm = np.conj(u-1j*v)/2   # clockwise circle 
    # amplitude and phase 
    wp_a = np.abs(wp)
    wm_a = np.abs(wm) 
    wp_p = np.angle(wp)
    wm_p = np.angle(wm) 
    # ellipse parameters 
    sema = wp_a + wm_a  # semi-major axis 
    semi = wp_a - wm_a  # semi-minor axis 
    ecc  = semi/sema
    pha  = 0.5*(wm_p - wp_p) # phase angle, angle at which velo reaches max
    inc  = 0.5*(wm_p + wp_p) # inclination, angle between semi-major axis and x-axis
    # convert to degrees for output 
    wp_p = 180*wp_p/np.pi
    wm_p = 180*wm_p/np.pi
    pha  = 180*pha/np.pi
    inc  = 180*inc/np.pi
    # goes from [-pi,0] to [pi,2pi] 
    try: # matrices
        wp_p[np.where(wp_p<0)] = wp_p[np.where(wp_p<0)] + 360
        wm_p[np.where(wm_p<0)] = wm_p[np.where(wm_p<0)] + 360
        pha[np.where(pha<0)]   = pha[np.where(pha<0)] + 360
        inc[np.where(inc<0)]   = inc[np.where(inc<0)] + 360
    except: # scalars 
        if wp_p<0: wp_p+=360
        if wm_p<0: wm_p+=360
        if pha<0:  pha+=360
        if inc<0:  inc+=360
        
    return sema,ecc,inc,pha  


def rot_uv(uu,vv,ang):
    """ rotate velocity field (uu,vv) by angle "ang" (radians) """
    return uu*np.cos(ang) - vv*np.sin(ang), uu*np.sin(ang) + vv*np.cos(ang)


def cmp2ap(re,im):
    """ convert complex to amplitude+phase """
    return np.abs(re+1j*im),np.arctan2(-im,re)*180/np.pi


def get_tpxo_on_grid(filenames,lonr,latr,return_ellipse=False,grang=None):
    """ read TPXO8 files (filenames=[ufile,hfile]) and interpolate it on lonr, latr grid
    if return_ellipse is True: ellipse components (SEMA, SEMI, INC, PHA -- angles in radian), 
    otherwise return amplitude, phase for u, v
    if grang != None, rotate field by angle grang """
    uname, hname = filenames

    nc = Dataset(uname,'r')
    latu = nc.variables['lat_u'][:]
    lonu = nc.variables['lon_u'][:]
    latv = nc.variables['lat_v'][:]
    lonv = nc.variables['lon_v'][:]
    nc.close()
    #lonu[lonu>180] -= 360
    #lonv[lonv>180] -= 360
    indxu, = np.where( (lonu>=(lonr%360).min()) & (lonu<=(lonr%360).max()) )
    indxv, = np.where( (lonv>=(lonr%360).min()) & (lonv<=(lonr%360).max()) )
    indyu, = np.where( (latu>=latr.min()) & (latu<=latr.max()) )
    indyv, = np.where( (latv>=latr.min()) & (latv<=latr.max()) )
    lonu = lonu[indxu]
    latu = latu[indyu]
    lonv = lonv[indxv]
    latv = latv[indyv]

    nc = Dataset(hname,'r')
    hu = nc.variables['hu'][indxu,indyu]
    hv = nc.variables['hv'][indxv,indyv]
    nc.close()

    nc = Dataset(uname,'r')
    ure = nc.variables['uRe'][indxu,indyu]*1e-4/hu    # cm²/s to m/s
    vre = nc.variables['vRe'][indxv,indyv]*1e-4/hv
    uim = nc.variables['uIm'][indxu,indyu]*1e-4/hu
    vim = nc.variables['vIm'][indxv,indyv]*1e-4/hv
    nc.close()
    ure[np.isnan(ure)] = 0.
    vre[np.isnan(vre)] = 0.
    uim[np.isnan(uim)] = 0.
    vim[np.isnan(vim)] = 0.

    ure = interp.RectBivariateSpline(lonu, latu, ure).ev(lonr%360,latr) # z.shape = (x.size, y.size)
    vre = interp.RectBivariateSpline(lonv, latv, vre).ev(lonr%360,latr)
    uim = interp.RectBivariateSpline(lonu, latu, uim).ev(lonr%360,latr)
    vim = interp.RectBivariateSpline(lonv, latv, vim).ev(lonr%360,latr)

    if return_ellipse:
        ua, up = cmp2ap(ure,uim)
        va, vp = cmp2ap(vre,vim)
        sema, ecc, inc, pha = ellipse(ua,up,va,vp)    # that is really sema, ecc, inc, pha
        inc, pha = np.deg2rad(inc), np.deg2rad(pha)
        if grang is not None:
            va -= grang
        return sema, ecc*sema, inc, pha
    else:
        if grang is not None:
            ure, vre = rot_uv(ure,vre,grang)
            uim, vim = rot_uv(uim, vim, grang)
        ua, up = cmp2ap(ure,uim)
        va, vp = cmp2ap(vre,vim)
        return ua, up, va, vp
 
