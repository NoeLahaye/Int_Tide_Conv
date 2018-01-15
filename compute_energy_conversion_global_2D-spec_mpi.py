'''
CV 2017/04/06 : I follow the 2D model of StLaurent and Garrett 2002
                to compute energy flux from barotropic to baroclinic energy on a regional scale 
                and make use of MPI library to parallelize the computation between subdomains. 
                to run the script : 
                > mpirun -np proc_number python compute_energy_conversion_global_2D-spec_mpi.py 
CV 2017/04/12 : For big domains, can specify N2 computed on subdomains 
                by changing lonmin_g to lonmin and so on. 
'''
import matplotlib     
matplotlib.use('Agg') # enables saving graphics 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['text.usetex'] = False
import scipy.interpolate as itp
import scipy.signal as sig
import scipy.stats as stats
from netCDF4 import Dataset
from SW_Density import SW_Density
from distance_sphere_matproof import dist_sphere_matproof
from convert_TPXO_to_ellipses import ellipse
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from change_coord import reproject_image_into_polar
from mpi4py import MPI 
clock  = datetime.now()

# --- MPI parameters --- 
npx,npy = 8,16 # number of processors in x and y directions (HPC-Wales : npx*npy must be a multiple of 16) 
#npx,npy = 1,1 # number of processors in x and y directions (HPC-Wales : npx*npy must be a multiple of 16) 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size!=npx*npy:
    exit('number of subdomains different from number of procs -> failure!')

# --- data location ---
path_data = '../../Data/'
#tname     = path_data+'WOA/woa13_decav_t00_01v2.nc'
#sname     = path_data+'WOA/woa13_decav_s00_01v2.nc'
tname     = path_data+'WOA/woa13_decav_t00_01v2_lon_000-360.nc' # HERE if lon in [-180,-090] or [090,180]
sname     = path_data+'WOA/woa13_decav_s00_01v2_lon_000-360.nc' # HERE if lon in [-180,-090] or [090,180]
uname     = path_data+'TPXO/u_tpxo8.nc' # see read_write_TPXO8_netcdf.py
#uname     = path_data+'TPXO/u_tpxo8_lon_-180-180.nc' # HERE if lon in [-090,090] 

# --- topography dataset --- 
#topo = 'etopo2' # dataset used : 'etopo2' or 'srtm30' 
topo = 'srtm30'

# --- global grid ---    
dspace            = 0.5         # resolution of the final map in degrees 
#lonmin_g,lonmax_g = -37,-34     # test box  
#latmin_g,latmax_g = 33,36  
#lonmin_g,lonmax_g = -53,-23    # RidgeMix cruise + Azores  
#latmin_g,latmax_g = 22,44
lonmin_g,lonmax_g = -180,-90     # quarter of earth  
latmin_g,latmax_g = -80,80

# --- output location and name --- 
path_write = '/scratch/clement.vic/Outputs/' 
file_write = 'Ef_'+topo+'_mpi_-180_-090_-80_80_%.3i.nc'%rank  # lonmin,lonmax,latmin,latmax 
#file_write = 'test_npts400_%.2i.nc'%rank  # lonmin,lonmax,latmin,latmax 

# --- miscellaneous --- 
if topo == 'etopo2':
    file_topo = path_data+'Etopo/ETOPO2v2c_f4.nc'
    #file_topo = path_data+'Etopo/ETOPO2v2c_f4_lon_000-360.nc' # HERE if lon in [-180,-090] or [090,180]
    npts      = 150  # half length of topo segments [number of grid points] 
elif topo == 'srtm30':
    file_topo = path_data+'SRTM30_PLUS/srtm30-plus_global.nc'
    npts      = 500  # half length of topo segments [number of grid points]  

nx     = int(0.7*2*npts) # that is the shape of the square actually used to compute spectrum 
                         # 0.7 is approx sqrt(2)/2 which is the max loss after rotation

deg        = 5                 # [degrees] half-width of the box in which N2 is computed 
zmin       = -500              # min depth to compute Ef [m], below contin. shelf roughly
g          = 9.81              # gravity [m s-2]
omega      = 7.2921e-5         # Earth's rotation rate [rad s-1]
M2         = 2.*np.pi/(44700.) # M2 tide frequency [rad s-1] 
lonm,latm  = -32.78,36.25      # RidgeMix mooring location 

# --- coordinates relative to proc number ---
lon1d_g       = np.arange(lonmin_g,lonmax_g+dspace,dspace) 
lat1d_g       = np.arange(latmin_g,latmax_g+dspace,dspace) 
nlat_g,nlon_g = lat1d_g.shape[0],lon1d_g.shape[0] 
di            = nlon_g/npx
dj            = nlat_g/npy
if di*npx<nlon_g: di+=1 # correction to make sure all the area is covered
if dj*npy<nlat_g: dj+=1
imin          = di*(rank%npx)
jmin          = dj*(rank//npx)
imax          = imin+di
jmax          = jmin+dj

# --- define subgrids ---
lon1d = lon1d_g[imin:imax] 
lat1d = lat1d_g[jmin:jmax] 
lon2d,lat2d = np.meshgrid(lon1d,lat1d)  
nlat,nlon = lon2d.shape
lonmin,lonmax = np.nanmin(lon2d),np.nanmax(lon2d)
latmin,latmax = np.nanmin(lat2d),np.nanmax(lat2d)

lonall = comm.gather([lonmin,lonmax],root=0)
latall = comm.gather([latmin,latmax],root=0)
dimall = comm.gather([nlat,nlon],root=0)
if rank==0:
    print ' Total domain : lon in [%i,%i], lat in [%i,%i]'%(lonmin_g,lonmax_g,latmin_g,latmax_g)
    for i in range(size):  
        print '  Processor %.3i will do lon in [%.2f,%.2f], lat in [%.2f,%.2f], [%i x %i] points'\
              %(i,lonall[i][0],lonall[i][1],latall[i][0],latall[i][1],dimall[i][1],dimall[i][0]) 

# ------ extract and interpolate TPXO ------------------------------
nc   = Dataset(uname,'r') # variables are on C-grid 
latu = nc.variables['lat_u'][:]
lonu = nc.variables['lon_u'][:] 
lonu[lonu>180] = lonu[lonu>180]-360. # HERE 
j = []; i = []
j.append(np.nanargmin(abs(latu[0,:]-latmin))-2) # +/-2 to make sure the ref region 
j.append(np.nanargmin(abs(latu[0,:]-latmax))+2) # is embedded in u and v grids
i.append(np.nanargmin(abs(lonu[:,0]-lonmin))-2)
i.append(np.nanargmin(abs(lonu[:,0]-lonmax))+2)

latu = nc.variables['lat_u'][i[0]:i[1],j[0]:j[1]]
lonu = nc.variables['lon_u'][i[0]:i[1],j[0]:j[1]]
lonu[lonu>180] = lonu[lonu>180]-360. # HERE 
latv = nc.variables['lat_v'][i[0]:i[1],j[0]:j[1]] 
lonv = nc.variables['lon_v'][i[0]:i[1],j[0]:j[1]]       
lonv[lonv>180] = lonv[lonv>180]-360. # HERE 

ua_tmp = nc.variables['ua'][0,i[0]:i[1],j[0]:j[1]] # a:amplitude, p:phase
up_tmp = nc.variables['up'][0,i[0]:i[1],j[0]:j[1]] # first index is 0==M2 
va_tmp = nc.variables['va'][0,i[0]:i[1],j[0]:j[1]]
vp_tmp = nc.variables['vp'][0,i[0]:i[1],j[0]:j[1]] 
nc.close()

# --- interpolate on reference lon,lat 
# RQ : itp.interp2d has issues that itp.griddata solve 
ua = itp.griddata((np.ravel(lonu),np.ravel(latu)),np.ravel(ua_tmp),(lon2d,lat2d))
up = itp.griddata((np.ravel(lonu),np.ravel(latu)),np.ravel(up_tmp),(lon2d,lat2d))
va = itp.griddata((np.ravel(lonv),np.ravel(latv)),np.ravel(va_tmp),(lon2d,lat2d))
vp = itp.griddata((np.ravel(lonv),np.ravel(latv)),np.ravel(vp_tmp),(lon2d,lat2d))
 
[sema,ecc,phi,pha] = ellipse(ua,up,va,vp) # phi is the angle in degrees between major axis and x-axis
ue  = sema*1e-2      # velocity in semi-major axis [m/s]
ve  = sema*ecc*1e-2  # velocity in semi-minor axis [m/s]
phi = phi*np.pi/180  # [rad] 

# ------ get mean topo on the grid, will be used to get N2b --------
nc   = Dataset(file_topo,'r') # etopo2 and srtm30 files have the same structure
#imin = np.nanargmin(abs(nc.variables['x'][:] - lonmin))-2
#imax = np.nanargmin(abs(nc.variables['x'][:] - lonmax))+2
imin = np.nanargmin(abs(nc.variables['x'][:]-360 - lonmin))-2 # HERE if lon in [-180,-090]
imax = np.nanargmin(abs(nc.variables['x'][:]-360 - lonmax))+2
jmin = np.nanargmin(abs(nc.variables['y'][:] - latmin))-2
jmax = np.nanargmin(abs(nc.variables['y'][:] - latmax))+2
#lonh = nc.variables['x'][imin:imax+1]
lonh = nc.variables['x'][imin:imax+1]-360  # HERE if lon in [-180,-090]
lath = nc.variables['y'][jmin:jmax+1]
h    = nc.variables['z'][jmin:jmax+1,imin:imax+1]
nc.close()
lonh,lath = np.meshgrid(lonh,lath)

hgrid = itp.griddata((np.ravel(lonh),np.ravel(lath)),np.ravel(h),(lon2d,lat2d))
hgrid[hgrid>0]     = 0     # land points  
hgrid[hgrid<-5500] = -5500 # no WOA data below

# ------ extract density from WOA then compute N2 ------------------
nc     = Dataset(tname,'r')
latwoa = nc.variables['lat'][:]
#lonwoa = nc.variables['lon'][:]
lonwoa = nc.variables['lon'][:]-360  # HERE if lon in [-180,-090]
z      = -nc.variables['depth'][:]
T      = nc.variables['t_an'][0,:,:,:]
nc.close()
nc     = Dataset(sname,'r')
S      = nc.variables['s_an'][0,:,:,:]
nc.close()

rho = SW_Density(T,S)
rho = np.array(rho)    # remove the mask included in WOA.  
rho[rho>1050] = np.nan # WOA fill_value is 1e36 
rho[rho==0]   = np.nan # rho=0 below seafloor
rho[rho<1000] = np.nan # rho=999.xxx below seafloor [depending on netCDF4 reading of nan...]

nzwoa,nywoa,nxwoa = rho.shape
dz      = np.transpose(np.tile(np.diff(z),(nywoa,nxwoa,1)),(2,0,1))
rho     = np.sort(rho,axis=0)                 # stable profile 
drho_dz = np.diff(rho,axis=0)/dz
zw      = 0.5*(z[1:]+z[:-1])
rho0    = np.nanmean(rho)
N2_tmp  = -(g/rho0)*drho_dz*(2*np.pi)**2      # has to be in [(rad s-1)^2]
nz      = N2_tmp.shape[0]
for k in np.arange(1,nz): # correct values == 0 (should not be any <0 as rho is sorted)
    for j in np.arange(nywoa):
        for i in np.arange(nxwoa):
            if N2_tmp[k,j,i]<=0: N2_tmp[k,j,i]=N2_tmp[k-1,j,i]

# --- Coriolis frequency [rad s-1] ---
f = 2*omega*np.sin(lat2d*np.pi/180.) 

# ====== BEGINNING OF REGIONAL LOOP ON LAT,LON ======================
#Ef     = np.zeros((nlat,nlon,nx,nx))  # energy flux (y,x,kh,theta) 
Ef_a   = np.zeros((nlat,nlon,nx))     # energy flux azimuthally averaged (y,x,kh) 
N2z    = np.zeros((nz,nlat,nlon))     # N2 (z,y,x) 
N2b    = np.zeros((nlat,nlon))        # N2 at the bottom (y,x) 
N0     = np.zeros((nlat,nlon))        # N0 parameter (y,x) 
b      = np.zeros((nlat,nlon))        # b parameter (y,x) 
Ef_t   = np.zeros((nlat,nlon))        # total energy flux (y,x) 
kh     = np.zeros((nlat,nlon,nx))     # horizontal wavenumber 
#theta  = np.zeros((nlat,nlon,nx))     # azimuth (anti-clockwise from major axis)  
for j in np.arange(nlat):
    clock_diff = datetime.now() - clock
    hour,sec = divmod(clock_diff.seconds,3600)
    hour     = hour + clock_diff.days*24
    minu,sec = divmod(sec,60)
    print ' ---> proc %.3i, time spent : %.2i h %.2i min %.2i sec, computation is at %.1f percent'\
          %(rank,hour,minu,sec,float(j)/nlat*100.) 
    for i in np.arange(nlon):
        nc   = Dataset(file_topo,'r') # etopo2 and srtm30 files have the same structure
        #ilon = np.nanargmin(abs(nc.variables['x'][:] - lon1d[i]))
        ilon = np.nanargmin(abs(nc.variables['x'][:]-360 - lon1d[i])) # HERE if lon in [-180,-090]
        ilat = np.nanargmin(abs(nc.variables['y'][:] - lat1d[j]))
        if nc.variables['z'][ilat,ilon]>zmin: # below continental shelf
            nc.close()
        else: 
            #lon  = nc.variables['x'][ilon-npts:ilon+npts] 
            lon  = nc.variables['x'][ilon-npts:ilon+npts]-360 # HERE if lon in [-180,-090]
            lat  = nc.variables['y'][ilat-npts:ilat+npts]
            h    = nc.variables['z'][ilat-npts:ilat+npts,ilon-npts:ilon+npts]
            nc.close()
            lon,lat = np.meshgrid(lon,lat)
            h[h>0] = 0 # land points 

            # --- first, compute distance between grid points 
            xx = dist_sphere_matproof(lat,lon,lat,lon1d[i])
            yy = dist_sphere_matproof(lat,lon,lat1d[j],lon)
            xx[lon<lon1d[i]] = -xx[lon<lon1d[i]] 
            yy[lat<lat1d[j]] = -yy[lat<lat1d[j]]
            dx = np.diff(xx,axis=1)
            dy = np.diff(yy,axis=0)

            # --- now sets a regular grid dxi=dyi=cst
            dxi   = np.nanmin((np.nanmin(dx),np.nanmin(dy))) # grid spacing for interpolation 
            dyi   = dxi
            xi    = np.arange(np.nanmin(xx),np.nanmax(xx),dxi)
            yi    = np.arange(np.nanmin(yy),np.nanmax(yy),dyi)
            xi,yi = np.meshgrid(xi,yi)
            hi    = itp.griddata((np.ravel(xx),np.ravel(yy)),np.ravel(h),
                                 (xi,yi),method='linear')

            # --- now rotate the grid to the major-axis direction
            #print '    angle of rotation [degrees, anticlockwise from x-axis] ',phi[j,i]*180/np.pi 
            xr    =   xi*np.cos(phi[j,i]) - yi*np.sin(phi[j,i])
            yr    =   xi*np.sin(phi[j,i]) + yi*np.cos(phi[j,i])
            hr    = itp.griddata((np.ravel(xi),np.ravel(yi)),np.ravel(hi),
                                 (xr,yr),method='linear')
     
            # --- reshape to get a square and remove nan at edges due to interpolation 
            nyr,nxr = hr.shape
            hr = hr[(nyr-nx)/2:-(nyr-nx)/2,(nxr-nx)/2:-(nxr-nx)/2]
            xr = xr[(nyr-nx)/2:-(nyr-nx)/2,(nxr-nx)/2:-(nxr-nx)/2]
            yr = yr[(nyr-nx)/2:-(nyr-nx)/2,(nxr-nx)/2:-(nxr-nx)/2]

            hi = hi[(nyr-nx)/2:-(nyr-nx)/2,(nxr-nx)/2:-(nxr-nx)/2] # for plotting purposes 
            xi = xi[(nyr-nx)/2:-(nyr-nx)/2,(nxr-nx)/2:-(nxr-nx)/2] 
            yi = yi[(nyr-nx)/2:-(nyr-nx)/2,(nxr-nx)/2:-(nxr-nx)/2]

            # --- compute 2D spectrum 
            kx = np.fft.fftshift(np.fft.fftfreq(nx,dxi))*2*np.pi # wavenumbers in x-direction = major axis
            ky = np.fft.fftshift(np.fft.fftfreq(nx,dyi))*2*np.pi # wavenumbers in y-direction = minor axis
            dk = kx[1]-kx[0]
 
            win_x   = np.tile(np.hanning(nx),(1,1))  # window before filtering 
            win_y   = np.tile(np.hanning(nx),(1,1)).T
            win     = np.dot(win_y,win_x)
            int_rec = nx**2             # integral of a squared rectangular window (as if no windowing) 
            int_win = np.nansum(win**2) # integral of the squared window 
            norm    = (int_rec/int_win)*1/(nx**2*dk**2) # [1/(rad m-1)^2] normalization constant 
            hr_win = (hr - np.nanmean(hr))*win # remove mean and window the signal 
            sp = norm*abs(np.fft.fftshift(np.fft.fft2(hr_win)))**2
            sp = sp*np.nanvar(hr)/np.sum(sp*dk*dk)
            kx2d,ky2d = np.meshgrid(kx,ky)
            sp[nx/2,nx/2]=np.nan # remove continuous component

            sp_polar, r, theta = reproject_image_into_polar(sp.T,origin=(nx/2,nx/2))
            kh[j,i,:] = r*dk # r is in pixel, multiply by dk to get wavenumber
            sp_polar[sp_polar==0]=np.nan
 
            weight = ( ue[j,i]**2*np.cos(theta)**2 
                     + ve[j,i]**2*np.sin(theta)**2 )
            gamma = np.zeros((nx,nx)) # temp variable 
            for k in np.arange(nx): 
                gamma[k,:] = sp_polar[k,:]*weight
            for t in np.arange(nx): 
                gamma[:,t] = gamma[:,t]*kh[j,i,:] 
           
            # --- compute N2(z) on a (2xdeg)x(2xdeg)-degree grid --- 
            imin = np.nanargmin(abs(lonwoa-lon2d[j,i]))-deg 
            imax = np.nanargmin(abs(lonwoa-lon2d[j,i]))+deg
            jmin = np.nanargmin(abs(latwoa-lat2d[j,i]))-deg
            jmax = np.nanargmin(abs(latwoa-lat2d[j,i]))+deg
            N2z[:,j,i] = np.nanmean(N2_tmp[:,jmin:jmax+1,imin:imax+1],axis=(1,2))

            # --- interpolate to get N2b and fit an exponential to N2 ---
            try:
                kmax     = np.nanargmax(np.sort(N2z[:,j,i])) # last index without nan        
                f_itp    = itp.interp1d(zw[:kmax+1],N2z[:kmax+1,j,i],fill_value='extrapolate')
                N2b[j,i] = f_itp(hgrid[j,i])
                slope,intercept,r_val,p_val,std_err = stats.linregress(zw[:kmax+1],np.log(N2z[:kmax+1,j,i]**0.5))
                N0[j,i]  = np.exp(intercept)/(2*np.pi)
                b[j,i]   = 1./slope
            except: 
                N2b[j,i] = np.nan
                N0[j,i]  = np.nan
                b[j,i]   = np.nan

            # --- compute Ef(K,theta) ---
            coef = 0.5*rho0*((N2b[j,i]-M2**2)*(M2**2-f[j,i]**2))**0.5/M2
            #Ef[j,i,:,:] = coef*gamma                                        # case 1  
            Ef = coef*gamma                                                  # case 2 

            # --- azimuthal integration [0,2pi] ---
            dtheta = theta[1] - theta[0] 
 
            #Ef_a = np.zeros(nx) # temp variable                             # case 1
            for k in np.arange(nx): 
                #Ef_a[k] = np.nansum(Ef[j,i,k,:]*kh[j,i,k]*dtheta)/(2*np.pi) # case 1 
                Ef_a[j,i,k] = np.nansum(Ef[k,:]*kh[j,i,k]*dtheta)/(2*np.pi)  # case 2 

            # --- equivalent mode number Eq (6) in StL and G 2002 ---  
            #kj = np.arange(nmodes)*np.pi/np.nanmean(-h_1d)*alpha # Eq (4) -> less robust  
            #kj[j,i,:] = np.arange(nmodes)*np.pi*(M2**2-f[j,i]**2)**0.5/(b[j,i]*N0[j,i]) # Eq (6)
            #dkj = kj[j,i,1] - kj[j,i,0] 
            k1 = np.pi*(M2**2-f[j,i]**2)**0.5/(b[j,i]*N0[j,i]) # Eq (6)
            dkj = k1 
            # kmin_int : min index over which performing integral
            try:    # ocean points
                kmin_int = np.nanargmin(abs(kh[j,i,:]-(k1-0.5*dkj)))+1 
            except: # land points
                kmin_int = -1

            # --- total energy flux ---  
            #Ef_t[j,i] = np.nansum(Ef_a[kmin_int:]*dk)                      # case 1  
            Ef_t[j,i] = np.nansum(Ef_a[j,i,kmin_int:]*dk)                   # case 2 

# ------ save energy flux in netcdf file ---------------------------
clock_diff = datetime.now() - clock
hour,sec = divmod(clock_diff.seconds,3600)
hour     = hour + clock_diff.days*24
minu,sec = divmod(sec,60)
print ' ===> proc %.3i, time spent : %.2i h %.2i min %.2i sec, save in netcdf file '\
      %(rank,hour,minu,sec) 
nc = Dataset(path_write+file_write,'w')
nc.createDimension('nz',zw.shape[0])
nc.createDimension('nlon',nlon)
nc.createDimension('nlat',nlat)
nc.createDimension('nkx',nx)
nc.createDimension('nky',nx)
nc.createDimension('nk',nx)
nc.createDimension('ntheta',nx)
#nc.createDimension('nmodes',nmodes)
nc.createVariable('z','f',('nz',))
nc.createVariable('N2z','f',('nz','nlat','nlon'))
nc.createVariable('lon','f',('nlat','nlon'))
nc.createVariable('lat','f',('nlat','nlon'))
nc.createVariable('h','f',('nlat','nlon'))
nc.createVariable('ue','f',('nlat','nlon'))
nc.createVariable('ve','f',('nlat','nlon'))
var = nc.createVariable('phi','f',('nlat','nlon'))
var.long_name = 'angle between ellipse major-axis and x-axis'
nc.createVariable('N2b','f',('nlat','nlon'))
nc.createVariable('N0','f',('nlat','nlon')) # works even if N0 and b are constant 
nc.createVariable('b','f',('nlat','nlon'))
nc.createVariable('f','f',('nlat','nlon'))
nc.createVariable('kh','f',('nlat','nlon','nk'))
var.long_name = 'equivalent mode number'
nc.createVariable('theta','f',('ntheta'))
#var = nc.createVariable('Ef','f',('nlat','nlon','nk','ntheta')) # case 1 
#var.long_name = 'Energy flux (lat,lon,K,theta)'                 # case 1 
var = nc.createVariable('Ef_a','f',('nlat','nlon','nk'))         # case 2 
var.long_name = 'Azimuthally-averaged energy flux (lat,lon,K)'   # case 2 
var = nc.createVariable('Ef_t','f',('nlat','nlon'))
var.long_name = 'Total energy flux (lat,lon)'
nc.variables['z'][:]      = zw
nc.variables['N2z'][:]    = N2z
nc.variables['lon'][:]    = lon2d
nc.variables['lat'][:]    = lat2d
nc.variables['h'][:]      = hgrid
nc.variables['ue'][:]     = ue
nc.variables['ve'][:]     = ve
nc.variables['phi'][:]    = phi
nc.variables['N2b'][:]    = N2b
nc.variables['N0'][:]     = N0
nc.variables['b'][:]      = b
nc.variables['f'][:]      = f
nc.variables['kh'][:]     = kh
try:
    nc.variables['theta'][:]  = theta # in case there is no ocean point, 
except:pass                           # in the tile theta is undefined  
#nc.variables['Ef'][:]     = Ef                                 # case 1 
nc.variables['Ef_a'][:]   = Ef_a                                # case 2  
nc.variables['Ef_t'][:]   = Ef_t
nc.close()

# --- delete variables to free memory for other tasks 
del zw,N2z,lon2d,lat2d,hgrid,ue,ve,phi,N2b,N0,b,f,kh,Ef_t,Ef_a
exit()

# ------ reconstruct global netcdf file ----------------------------
# WAIT FOR EVERY PROC TO HAVE WRITTEN 

path_write = '/scratch/clement.vic/Outputs/' 
fname      = 'Ef_srtm30_mpi_000_090_-80_80_000.nc'
npx,npy    = 8,16
size       = npx*npy

# --- extract dimensions --- 
rank = 0 
nc0 = Dataset(path_write+fname[:-6]+'%.3i.nc'%rank,'r') 
nlat0,nlon0,nx = nc0.variables['kh'][:].shape 
nc0.close()

nlat,nlon = 0,0 
for rank in np.arange(npx): 
    nc0 = Dataset(path_write+fname[:-6]+'%.3i.nc'%rank,'r') 
    nlon += nc0.variables['kh'][:].shape[1] 
nc0.close()

for rank in np.arange(npy): 
    nc0 = Dataset(path_write+fname[:-6]+'%.3i.nc'%rank,'r') 
    nlat += nc0.variables['kh'][:].shape[0] 
nc0.close()


# --- create global file --- 
print ' ... create global file ... ' 
nc = Dataset(path_write+fname[:-7]+'.nc','w')
nc.createDimension('nlon',nlon)
nc.createDimension('nlat',nlat)
nc.createDimension('nkx',nx)
nc.createDimension('nky',nx)
nc.createDimension('nk',nx)
nc.createDimension('ntheta',nx)
nc.createVariable('lon','f',('nlat','nlon'))
nc.createVariable('lat','f',('nlat','nlon'))
nc.createVariable('h','f',('nlat','nlon'))
nc.createVariable('ue','f',('nlat','nlon'))
nc.createVariable('ve','f',('nlat','nlon'))
var = nc.createVariable('phi','f',('nlat','nlon'))  
var.long_name = 'angle between ellipse major-axis and x-axis' 
nc.createVariable('N2b','f',('nlat','nlon'))
nc.createVariable('N0','f',('nlat','nlon'))
nc.createVariable('b','f',('nlat','nlon'))
nc.createVariable('f','f',('nlat','nlon'))
nc.createVariable('kh','f',('nlat','nlon','nk'))
var.long_name = 'equivalent mode number'
#nc.createVariable('theta','f',('nlat','nlon','ntheta'))  # case 1 
nc.createVariable('theta','f',('ntheta'))                 # case 2 
#var = nc.createVariable('Ef','f',('nlat','nlon','nk','ntheta'))  # case 1 
#var.long_name = 'Energy flux (lat,lon,K,theta)'                  # case 1 
var = nc.createVariable('Ef_a','f',('nlat','nlon','nk'))          # case 2 
var.long_name = 'Azimuthally-averaged energy flux (lat,lon,K)'    # case 2
var = nc.createVariable('Ef_t','f',('nlat','nlon'))
var.long_name = 'Total energy flux (lat,lon)'
for rank in range(size):
    print ' -> copy file %.2i in global file '%rank
    ncr = Dataset(path_write+fname[:-6]+'%.3i.nc'%rank,'r') 
    nlatr,nlonr = ncr.variables['h'][:].shape
    imin = nlon0*(rank%npx)
    jmin = nlat0*(rank//npx)
    imax = imin+nlonr
    jmax = jmin+nlatr
    nc.variables['lon'][jmin:jmax,imin:imax]       = ncr.variables['lon'][:] 
    nc.variables['lat'][jmin:jmax,imin:imax]       = ncr.variables['lat'][:] 
    nc.variables['h'][jmin:jmax,imin:imax]         = ncr.variables['h'][:] 
    nc.variables['ue'][jmin:jmax,imin:imax]        = ncr.variables['ue'][:] 
    nc.variables['ve'][jmin:jmax,imin:imax]        = ncr.variables['ve'][:] 
    nc.variables['phi'][jmin:jmax,imin:imax]       = ncr.variables['phi'][:] 
    nc.variables['N2b'][jmin:jmax,imin:imax]       = ncr.variables['N2b'][:] 
    nc.variables['N0'][jmin:jmax,imin:imax]        = ncr.variables['N0'][:] 
    nc.variables['b'][jmin:jmax,imin:imax]         = ncr.variables['b'][:] 
    nc.variables['f'][jmin:jmax,imin:imax]         = ncr.variables['f'][:] 
    nc.variables['kh'][jmin:jmax,imin:imax,:]      = ncr.variables['kh'][:] 
    #nc.variables['theta'][jmin:jmax,imin:imax,:]   = ncr.variables['theta'][:] # case 1 
    #nc.variables['Ef'][jmin:jmax,imin:imax,:,:]    = ncr.variables['Ef'][:]    # case 1 
    nc.variables['Ef_a'][jmin:jmax,imin:imax,:]    = ncr.variables['Ef_a'][:]   # case 2 
    nc.variables['Ef_t'][jmin:jmax,imin:imax]      = ncr.variables['Ef_t'][:] 
    print 'end of subdomain file'
    if rank==0 : 
        nc.variables['theta'][:] = ncr.variables['theta'][:] 
    ncr.close()
nc.close()

