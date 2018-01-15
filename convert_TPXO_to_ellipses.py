'''
CV 2016/10/21 : compute ellipses parameters given the amplitude and phase of u,v
                adapted from matlab's function by Zhigang Xu [see also croco tools]
                _a : amplitude [degrees, not radians]  
                _p : phase     [degrees, not radians]  
'''
import numpy as np

def ellipse(u_a,u_p,v_a,v_p):
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
 
