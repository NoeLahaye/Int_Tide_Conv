import numpy as np
Erad = 6371e3   # Earth radius
def dist_sphere_matproof(lat0,lon0,lat1,lon1):
    # --- http://www.movable-type.co.uk/scripts/latlong.html --- 
    lat0 = lat0*np.pi/180.
    lat1 = lat1*np.pi/180.
    lon0 = lon0*np.pi/180.
    lon1 = lon1*np.pi/180.
    aa  = np.sin(0.5*(lat1-lat0))**2 + np.cos(lat1)*np.cos(lat0)*np.sin(0.5*(lon1-lon0))**2
    dd  = 2*np.arctan2(aa**0.5,(1-aa)**0.5)*Erad
    return dd

def dx_deg2sphere(lon,lat,interp_back=False):
    ''' array have shape (Ny,Nx) '''
    dx = np.diff(lon,axis=1)*np.pi/180
    dy = np.diff(lat,axis=0)*np.pi/180
    dy = dy/Erad
    if interp_back:
        dy = 0.5*(dy[1:,:] + dy[:-1,:])
        dx = 0.5*(dx[:,1:] + dx[:,:-1])/Erad/np.sin(lat[:,1:-1]*np.pi/180.)
    else:
        dx = dx/Erad/np.sin(0,5*(lat[:,1:]+lat[:,:-1])*np.pi/180.)
    return dx, dy