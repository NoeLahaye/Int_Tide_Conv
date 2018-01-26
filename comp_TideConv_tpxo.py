
# coding: utf-8

# # comp_TideConv_tpxo
# compute barotropic to baroclinic tide conversion using [St. Laurent & Garrett 2002], adapted from C. Vic code.
# 
# ### This code:
# * regional scale: grid is regular and cartesioen (approximately)
# * stratification is horizontally homogeneous
# * tides are taken from TPXO8 or forcing file of ROMS computation
# * topography, longitude and latitude are retrieved from ROMS grid file
# 
# ### Limitations:
# Originally, [St. Laurent & Garrett 2002] is relevant for "weak" topography and no important variations of tide and bathymetry. Here, we implicetely assume that tidal constituents are "frozen" at every points. Besides, finite-depth effect are taken into account by using a finite lower bound of integration in the spectral space (roughly corresponding to mode 1), which is taken constant here (whereas variations of kh_1 scales like 1/H)
# 

import numpy as np
import scipy.interpolate as interp
#import scipy.signal as sig
import scipy.stats as stats
from netCDF4 import Dataset
from datetime import datetime
from mpi4py import MPI 
#from SW_Density import SW_Density as rhop # temporary
from comp_rho import rhop
#from distance_sphere_matproof import dist_sphere_matproof
from change_coord import reproject_image_into_polar
#from pad_coords import pad_coords   # for special padding outside of domain
from detrend_2d import detrend_2d
from convert_TPXO_to_ellipses import get_tpxo_on_grid

clock  = datetime.now()
comm = MPI.COMM_WORLD

doverb = True

# --- MPI parameters --- 
npx,npy = 2,2 # number of processors in x and y directions  
#npx,npy = 1,1 # number of processors in x and y directions  

rank = comm.Get_rank()
size = comm.Get_size()
#rank, size = 0, 1
if size != npx*npy:
    print('mpi size is {0} vs. {1}x{2}'.format(size,npx,npy))
    raise ValueError('number of subdomains different from number of procs -> failure!')

# --- data location ---
path_data = '/data0/project/vortex/lahaye/Tide_Conv/input_data/' # './input_data/' #
#path_data = '/net/krypton'+path_data     # if not on LOPS hub (or libra)
path_alpha = '/net/alpha/exports/sciences/data/REFERENCE_DATA/TIDES/'

# --- climato ---
clim = "lucky"
if clim == "lucky":
    cname = path_data+"lucky_ts_meanSODA_winter.nc"

# --- topography dataset --- 
topo = "lucky"

# --- tide dataset --- 
tide = 'tpxo8'
if tide == 'lucky':
    collot = True    # tides and topo defined on same points
    uname = path_data+'luckym2_frc.nc' # see read_write_TPXO8_netcdf.py
elif tide == 'tpxo8':
    collot = False
    uname = path_alpha+'TPXO8/uv.m2_tpxo8_atlas_30c_v1.nc'
    hname = path_alpha+'TPXO8/grid_tpxo8_atlas_30c_v1.nc'
    
# --- global grid ---   
colloc = True    # use same grid as topo // False not implemented
if colloc:
    nstep = 4
else:
    dspace = 1e3       # resolution of the final map in meter
#lonmin_g,lonmax_g = -33,-32     # test box // None takes the whole domain defined by h  
#latmin_g,latmax_g = 37,37.7  
lonmin_g, latmin_g, lonmax_g, latmax_g = [None]*4  # entire domain

# --- output location and name --- 
path_write = path_data.replace('input','output')
file_write = 'Ef_{0}_{1}_mpi.{2:02d}.nc'.format(topo,tide,rank)   

# --- miscellaneous --- 
if topo == "lucky":
    file_topo = path_data+"lucky_grd.nc"
    Lchk = 150e3    # length of window for computing topo spectrum
    varx, vary = 'lon_rho', 'lat_rho'    # variable names in netCDF files
    varh = 'h'

zmin       = -100              # min depth to compute Ef [m], below contin. shelf roughly
g          = 9.81              # gravity [m s-2]
omega      = 7.2921e-5         # Earth's rotation rate [rad s-1]
M2         = 2.*np.pi/(44700.) # M2 tide frequency [rad s-1] 
lonm,latm  = -32.16,37.17      # Momar mooring location 
Erad = 6371e3                  # Earth radius [m]

### warning: make sure the following lines are in agreement with subsequent parameters and grids
nxout = 256    # number of points for k modulus
nxoth = 128    # number of points for k angle
khout = np.linspace(1./Lchk,1./750/np.sqrt(2.),nxout)*np.pi  # k adim for output = k*bar(N)*H/sqrt(M2^2-f^2)
thout = np.linspace(0,np.pi*2,nxoth+1)[:-1]  # theta for output


# ------ get mean topo on the grid, will be used to get N2b --------
# load the entire grid (regional modelling purpose)
nc   = Dataset(file_topo,'r') # etopo2 and srtm30 files have the same structure
if topo == 'lucky':
    lon_h = nc.variables[varx][:].T
    lat_h = nc.variables[vary][:].T
    nlon_h, nlat_h = lon_h.shape
    if lonmin_g is None:
        ix = [0]
    else:
        ix = [np.abs(lon_h-lonmin_g).argmin(axis=0).min()]
    if lonmax_g is None:
        ix.append(nlon_h)
    else:
        ix.append(np.abs(lon_h-lonmax_g).argmin(axis=0).max())
    if latmin_g is None:
        jy = [0]
    else:
        jy = [np.abs(lat_h-latmin_g).argmin(axis=1).min()]
    if latmax_g is None:
        jy.append(nlat_h)
    else:
        jy.append(np.abs(lat_h-latmax_g).argmin(axis=1).max())
    h_t = -nc.variables[varh][jy[0]:jy[1],ix[0]:ix[1]].T
    h_t[h_t>=-2.5]     = 0     # land points  
    #lon = lon[ix[0]:ix[1],jy[0]:jy[1]]
    #lat = lat[ix[0]:ix[1],jy[0]:jy[1]]
    #Nxh, Nyh = lon_g.shape
    dx_h = 1/nc.variables['pm'][:].T
    dy_h = 1/nc.variables['pn'][:].T
nc.close()


# --- grids (coordinates relative to proc number) ---
# working with (x,y)-ordered grids
if colloc:
    lon2d_g = lon_h[ix[0]:ix[1]:nstep,jy[0]:jy[1]:nstep]
    lat2d_g = lat_h[ix[0]:ix[1]:nstep,jy[0]:jy[1]:nstep]
    nlon_g, nlat_g = lon2d_g.shape
else:
    lon1d_g       = np.arange(lonmin_g,lonmax_g+dspace,dspace) 
    lat1d_g       = np.arange(latmin_g,latmax_g+dspace,dspace) 
    nlat_g,nlon_g = lat1d_g.shape[0],lon1d_g.shape[0] 
di            = nlon_g//npx
dj            = nlat_g//npy
if di*npx<nlon_g: di+=1 # correction to make sure all the area is covered
if dj*npy<nlat_g: dj+=1
imin          = di*(rank%npx)
jmin          = dj*(rank//npx)
imax          = imin+di
jmax          = jmin+dj

# --- define subgrids ---
if colloc:
    lon2d, lat2d = lon2d_g[imin:imax,jmin:jmax], lat2d_g[imin:imax,jmin:jmax]
else:
    lon1d = lon1d_g[imin:imax] 
    lat1d = lat1d_g[jmin:jmax] 
    lon2d,lat2d = np.meshgrid(lon1d,lat1d)  
nlon,nlat = lon2d.shape
lonmin,lonmax = np.nanmin(lon2d),np.nanmax(lon2d)
latmin,latmax = np.nanmin(lat2d),np.nanmax(lat2d)

if size > 1:
    lonall = comm.gather([lonmin,lonmax],root=0)
    latall = comm.gather([latmin,latmax],root=0)
    dimall = comm.gather([nlat,nlon],root=0)
else:
    lonall = [[lonmin,lonmax]]
    latall = [[latmin,latmax]]
    dimall = [[nlat,nlon]]
    
if rank==0:
    print('Total domain : lon in [%.1f,%.1f], lat in [%.1f,%.1f]'          %(lon2d_g.min(),lon2d_g.max(),lat2d_g.min(),lat2d_g.max()))
    for i in range(size):  
        print(' Processor %.3i will do lon in [%.1f,%.1f], lat in [%.1f,%.1f], [%i x %i] points'              %(i,lonall[i][0],lonall[i][1],latall[i][0],latall[i][1],dimall[i][1],dimall[i][0]) )

# load grid angle for rotation to ellipse frame since we do not interpolate field on regular grid here        
if colloc:
    nc = Dataset(file_topo,'r')
    hgrid = -nc.variables[varh][jy[0]:jy[1]:nstep,ix[0]:ix[1]:nstep].T[imin:imax,jmin:jmax]
    hgrid[hgrid>=-2.5] = 0.
    angrid = nc.variables['angle'][jy[0]:jy[1]:nstep,ix[0]:ix[1]:nstep].T[imin:imax,jmin:jmax]
    nc.close()
else:
    raise ValueError('colloc False not implementend for hgrid: need to interpolate')
    
# --- Coriolis frequency [rad s-1] ---
f = 2*omega*np.sin(lat2d*np.pi/180.) 

# ------ extract Tides ------------------------------

if tide=='lucky' and collot and colloc:
    nc   = Dataset(uname,'r')
    phi = np.deg2rad(nc.variables['tide_Cangle'][0,...].T[ix[0]:ix[1]:nstep,jy[0]:jy[1]:nstep][imin:imax,jmin:jmax])
    # N.B.: maybe the sign is wrong (I assume count positive cc-wise with E as origin)
    pha = np.deg2rad(nc.variables['tide_Cphase'][0,...].T[ix[0]:ix[1]:nstep,jy[0]:jy[1]:nstep][imin:imax,jmin:jmax])
    ue = nc.variables['tide_Cmax'][0,...].T[ix[0]:ix[1]:nstep,jy[0]:jy[1]:nstep][imin:imax,jmin:jmax]
    ve = nc.variables['tide_Cmin'][0,...].T[ix[0]:ix[1]:nstep,jy[0]:jy[1]:nstep][imin:imax,jmin:jmax]
    nc.close()
elif tide=='tpxo8':
    ue, ve, phi, pha = get_tpxo_on_grid([uname,hname],lon2d,lat2d,return_ellipse=True)

phi = np.deg2rad(phi)  # angle between major axis and East  [rad] (beware sign)

# ------ extract density profile, compute N2 (horizontally homogeneous) ------------------
if clim == "lucky":
    nc = Dataset(cname,'r')
    T = nc.variables['temp_roms_avg'][:]
    S = nc.variables['salt_roms_avg'][:]
    zz = nc.variables['depth'][:]
    nz = zz.size
    
rho = np.sort(rhop(T,S)) #SW_Density(T,S) # sorting is cheating here
rho0 = rho.mean()
frho = interp.pchip(zz[::-1],rho[::-1],extrapolate=True)
N2_tmp = -(g/rho0)*frho.derivative()(zz)    # # has to be in [(rad s-1)^2] (-> *(2pi)**2 ? nop)
# temporary fixing:
if N2_tmp[-1]==0: N2_tmp[-1] = 1e-8
indneg, = np.where(N2_tmp<=0.)
for ii in indneg:
    N2_tmp[ii] = (N2_tmp[ii-1] + N2_tmp[ii+1])/2
fN2 = interp.pchip(zz[::-1],N2_tmp,extrapolate=True)    


# fit exponential profile
slope,intercept,r_val,p_val,std_err = stats.linregress(zz,np.log(N2_tmp**0.5))
N0  = np.exp(intercept)
b   = 1./slope

if doverb:
    if indneg.size>0:
        print('had to resort stratif for {} values'.format(indneg.size))
    print('exponential interpolation for stratification: N0={0}, b={1}'.format(N0,b))

# ------ prepare netcdf file and store "fix" variables---------------------------

ncw = Dataset(path_write+file_write,'w')

# global attributes
ncw.created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ncw.generating_script = sys.argv[0]

ncw.createDimension('z',zz.size)
ncw.createDimension('lon',nlon)
ncw.createDimension('lat',nlat)
ncw.createDimension('kh',nxout)
ncw.createDimension('theta',nxoth)
#ncw.createDimension('nmodes',nmodes)
ncvar = ncw.createVariable('z','f',('z',))
ncvar.long_name = "depth"
ncvar.properties = "algebraic, negative downward, 0 at surface"
ncvar.units  = "m"
ncvar = ncw.createVariable('N2z','f',('z'))
ncvar.units = "s^{-2}"
ncvar.long_name = "Brunt-Vaisala frequency squared"
ncvar.properties = "vertical profile"
ncvar = ncw.createVariable('lon','f',('lat','lon'))
ncvar.units = "degres E"
ncvar.long_name = "longitude"
ncvar = ncw.createVariable('lat','f',('lat','lon'))
ncvar.units = "degres N"
ncvar.long_name = "latitude"
ncvar = ncw.createVariable('h','f',('lat','lon'))
ncvar.units = "m"
ncvar.long_name = "depth of bottom"
ncvar.properties = "negative"
ncvar = ncw.createVariable('ue','f',('lat','lon'))
ncvar.units = 'm/s'
ncvar.long_name = "major tidal ellipse amplitude"
ncw.createVariable('ve','f',('lat','lon'))
ncvar.units = 'm/s'
ncvar.long_name = "minor tidal ellipse amplitude"
var = ncw.createVariable('phi','f',('lat','lon'))
var.long_name = 'angle between ellipse major-axis and E-axis'
var.units = 'deg'
var.properties = "positive counterclockwise with East-axis as origin"
var = ncw.createVariable('lambda1','f',('lat','lon'))
var.long_name = 'equivalent mode 1 horizontal wavelength'
var.units = 'km'
var = ncw.createVariable('N2b','f',('lat','lon'))
var.units = "s^{-2}"
var.long_name = "N^2 at bottom"
var = ncw.createVariable('N0','f',()) # works even if N0 and b are constant 
var.units = "s^{-1}"
var.long_name = "Exponential fit of N(z)=N0.exp(-z/b)"
ncw.createVariable('b','f',())
var = ncw.createVariable('f','f',('lat','lon'))
var.units = "rad/s"
var.long_name = "Coriolis frequency"
ncw.createVariable('kh','f',('kh'))
var.units = "rad/m"
var.long_name = 'equivalent mode number'
var = ncw.createVariable('theta','f',('theta'))
var.longname = "wave-vector angle"
var.units = "rad"
var = ncw.createVariable('Ef','f',('lat','lon','kh','theta')) # case 1 
var.units = "W"
var.long_name = 'Vertical energy flux spectral density'                 # case 1 
var = ncw.createVariable('Ef_a','f',('lat','lon','kh'))         # case 2 
var.long_name = 'Azimuthally-averaged spectral density of vertical energy flux'   # case 2 
var.units = "W/m"
var = ncw.createVariable('Ef_t','f',('lat','lon'))
var.long_name = 'Vertical energy flux'
var.units = "W/m^2"
var = ncw.createVariable('h_sp','f',('lat','lon','kh','theta'))
var.long_name = 'Normalized local spectrum'
var.units = "m^4"

ncw.variables['z'][:]      = zz
ncw.variables['N2z'][:]    = N2_tmp
ncw.variables['lon'][:]    = lon2d.T
ncw.variables['lat'][:]    = lat2d.T
ncw.variables['h'][:]      = hgrid.T
ncw.variables['ue'][:]     = ue.T
ncw.variables['ve'][:]     = ve.T
ncw.variables['phi'][:]    = np.rad2deg(phi.T)
ncw.variables['N0'][:]     = N0
ncw.variables['b'][:]      = b
ncw.variables['f'][:]      = f.T
ncw.variables['kh'][:]     = khout
ncw.variables['theta'][:]  = thout 

ncvar = ncw.variables


# ====== BEGINNING OF REGIONAL LOOP ON LAT,LON ======================

for j in range(nlat): #
    clock_diff = datetime.now() - clock
    hour,sec = divmod(clock_diff.seconds,3600)
    hour     = hour + clock_diff.days*24
    minu,sec = divmod(sec,60)
    print( ' ---> proc %.3i, time spent : %.2i h %.2i min %.2i sec, computation is at %.1f percent'          %(rank,hour,minu,sec,float(j)/nlat*100.)) 
    for i in range(nlon): #
        xpos, ypos = lon2d[i,j], lat2d[i,j]
        ix, jy = np.unravel_index( ((lon_h-xpos)**2 + (lat_h-ypos)**2).argmin() , (nlon_h,nlat_h))
        xx = np.cumsum(dx_h[:,jy]) - dx_h[:ix+1,jy].sum()    # substracte midle point
        yy = np.cumsum(dy_h[ix,:]) - dy_h[ix,:jy+1].sum()
        #degy = Lchk/Erad*180./np.pi
        #degx = degy/np.cos(ypos*np.pi/180.)
        # endpoints
        #il = np.abs(lon_h[:,jy]-xpos+degx).argmin()-2
        #ir = np.abs(lon_h[:,jy]-xpos-degx).argmin()+2
        #jb = np.abs(lat_h[ix,:]-ypos+degy).argmin()-2
        #jt = np.abs(lat_h[ix,:]-ypos-degy).argmin()+2
        il = np.abs(xx+Lchk).argmin() - 2    # indices in global (topo) full-resolution grid
        ir = np.abs(xx-Lchk).argmin() + 2
        jb = np.abs(yy+Lchk).argmin() - 2
        jt = np.abs(yy-Lchk).argmin() + 2
        
        indx = slice(max(il,0),min(ir,nlon_h))
        indy = slice(max(jb,0),min(jt,nlat_h))
        h = h_t[indy,indx].T   
        lon = lon_h[indx,indy]
        lat = lat_h[indx,indy]
        xx, yy = xx[indx], yy[indy]
        # approximation: take mean cell size to compute wavenumbers
        dxi = 0.5*(dx_h[indy,indx].mean() + dy_h[indy,indx].mean())
        #dxi = 0.5*(np.diff(lon,axis=0).mean()*np.cos(ypos*np.pi/180.) \
        #            + np.diff(lat,axis=1).mean())*Erad*np.pi/180.
        
        # do we need to pad ?
        npadl = npadr = npadt = npadb = 0
        if -Lchk < xx.min():
            if il > 0: print('problem')
            npadl = int((xx.min() + Lchk)/dxi) + 1
        if xx.max() < Lchk:
            if ir < nlon_h: print('problem')
            npadr = int((Lchk - xx.max())/dxi) + 1
        if -Lchk < yy.min():
            if jb > 0: print('problem')
            npadb = int((yy.min() + Lchk)/dxi) + 1
        if yy.max() < Lchk:
            if jt < nlat_h: print('problem')
            npadt = int((Lchk - yy.max())/dxi) + 1

        if max(npadl,npadr,npadt,npadb) > 0:
            h = np.pad(h,((npadl,npadr),(npadb,npadt)),'edge')
        # make it square (again)
        dn = h.shape[1] - h.shape[0]
        if dn < 0:
            h = h[(-dn)//2:-((-dn)//2+(-dn)%2),:]
        elif dn > 0:
            h = h[:,dn//2:-(dn//2+dn%2)]
        if np.abs(dn) > 3: print('squaring {0} at i={1}, j={2}'.format(np.abs(dn),i,j))
        nx, ny = h.shape
        if nx != ny: print('problem still not square')
            
        kx = np.fft.fftshift(np.fft.fftfreq(nx,dxi))*2*np.pi # wavenumbers in x-direction = major axis
        dk = kx[1]-kx[0]
        h = detrend_2d(h) # apply bilinear detrend
        win_x   = np.tile(np.hanning(nx),(1,1))  # window before fft-ing 
        win     = np.dot(win_x.T,win_x)
        int_rec = nx**2             # integral of a squared rectangular window (as if no windowing) 
        int_win = np.nansum(win**2) # integral of the squared window 
        norm    = (int_rec/int_win)*1/(nx**2*dk**2) # [1/(rad m-1)^2] normalization constant 
        sp = norm*abs(np.fft.fftshift(np.fft.fft2(h*win)))**2
        sp = sp*np.nanvar(h)/np.sum(sp*dk*dk)
        kx2d,ky2d = np.meshgrid(kx,kx)
        sp[np.where(np.logical_and(kx2d==0,ky2d==0))] = np.nan # remove continuous component
        
        sp_polar, r, theta = reproject_image_into_polar(sp.T,origin=(nx//2,nx//2),theta_shift=(phi[i,j]+angrid[i,j]))
        kh = r*dk # r is in pixel, multiply by dk to get wavenumber
        sp_polar[r==0] = np.nan

        weight = ( ue[i,j]**2*np.cos(theta)**2 + ve[i,j]**2*np.sin(theta)**2 )
        gamma = sp_polar*weight[None,:]*kh[:,None]
            
        N2b = fN2(hgrid[i,j])
        
        # --- compute Ef(K,theta) ---
        coef = 0.5*rho0*((N2b-M2**2)*(M2**2-f[i,j]**2))**0.5/M2 
        Ef = coef*gamma 

        # --- azimuthal integration [0,2pi] ---
        dtheta = theta[1] - theta[0] 
        Ef_a = np.nansum(Ef*dtheta,axis=1)*kh/(2*np.pi)

        # --- equivalent mode number Eq (6) in StL and G 2002 ---  
        #k1 = np.pi*(M2**2-f[i,j]**2)**0.5/(b*N0) # Eq (6), H neglected
        k1 = np.pi*np.sqrt(M2**2-f[i,j])/(b*N0*(1-np.exp(hgrid[i,j]/b)))
        dkj = k1 
        
        try:   
            kmin_int = np.nanargmin(abs(kh-(k1-0.5*dkj)))+1 # min index over which performing integral
        except: # land points
            kmin_int = -1
        
        # store in netCDF file NRJ fluxes
        ncvar['Ef'][j,i,:,:] = interp.RectBivariateSpline(kh,theta,Ef,kx=1,ky=1)(khout,thout)  # energy flux sp. density
        ncvar['Ef_a'][j,i,:] = interp.pchip(kh,Ef_a)(khout)     # NRJ flux azimuth.-averaged
        ncvar['Ef_t'][j,i] = np.nansum(Ef_a[kmin_int:]*dk) # total NRJ flux
        ncvar['h_sp'][j,i,:,:] = interp.RectBivariateSpline(kh,theta,sp_polar,kx=1,ky=1)(khout,thout) # spectrum of topography
        ncvar['N2b'][j,i]    = N2b
        ncvar['lambda1'][j,i] = 2*np.pi/k1/1e3
# ------ end of loop: print timing at the beginning ---------------------------
ncw.close()

clock_diff = datetime.now() - clock
hour,sec = divmod(clock_diff.seconds,3600)
hour     = hour + clock_diff.days*24
minu,sec = divmod(sec,60)
print(' ===> proc %.3i, time spent : %.2i h %.2i min %.2i sec'%(rank,hour,minu,sec)) 

