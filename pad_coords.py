import numpy as np

def pad_coords(lon,lat,npads):
    ''' pad outside lon, lat by npads points (order west, top, east, south)
    lon, lat ar longitude, latitude (or whatever) and have size (Nx, Ny)
    extend domain with constant grid increment '''
    
    npadl, npadt, npadr, npadb = npads
    nchx, nchy = lon.shape
    
    ### first compute grid increments (derivative of position) at edges
    if npadl > 0:
        dxl = np.diff(lon[:2,:],axis=0)
        dyl = np.diff(lat[:2,:],axis=0)
    if npadr > 0:
        dxr = np.diff(lon[-2:,:],axis=0)
        dyr = np.diff(lat[-2:,:],axis=0)
    if npadb > 0:
        dyb = np.diff(lat[:,:2],axis=1)
        dxb = np.diff(lon[:,:2],axis=1)
    if npadt > 0:
        dyt = np.diff(lat[:,-2:],axis=1)
        dxt = np.diff(lon[:,-2:],axis=1)
        
    ### Next compute padding blocks (8 in total, 1 per edge + 1 per corner)
    if npadl > 0:
        lblon = lon[:1,:] + dxl*np.arange(-npadl,0)[:,None]
        lblat = lat[:1,:] + dyl*np.arange(-npadl,0)[:,None]
    else:
        lblon, lblat = np.zeros((npadl,nchy)), np.zeros((npadl,nchy))
    if npadb > 0:
        bblon = lon[:,:1] + dxb*np.arange(-npadb,0)[None,:]
        bblat = lat[:,:1] + dyb*np.arange(-npadb,0)[None,:]
    else:
        bblon, bblat = np.zeros((nchx,npadb)), np.zeros((nchx,npadb))
    if npadr > 0:
        rblon = lon[-1:,:] + dxr*np.arange(1,npadr+1)[:,None]
        rblat = lat[-1:,:] + dyr*np.arange(1,npadr+1)[:,None]
    else:
        rblon, rblat = np.zeros((npadr,nchy)), np.zeros((npadr,nchy))
    if npadt > 0:
        tblon = lon[:,-1:] + dxt*np.arange(1,npadt+1)[None,:]
        tblat = lat[:,-1:] + dyt*np.arange(1,npadt+1)[None,:]
    else:
        tblon, tblat = np.zeros((nchx,npadb)), np.zeros((nchx,npadb))
    if npadl > 0 and npadb > 0:
        lbblon = lon[0,0] + dxl.squeeze()[0]*np.tile(np.arange(-npadl,0)[:,None],(1,npadb))
        lbblat = lat[0,0] + dyb.squeeze()[0]*np.tile(np.arange(-npadb,0)[None,:],(npadl,1))
    else:
        lbblon, lbblat = np.zeros((npadl,npadb)), np.zeros((npadl,npadb))
    if npadr > 0 and npadb > 0:
        rbblon = lon[-1,0] + dxr.squeeze()[0]*np.tile(np.arange(1,npadr+1)[:,None],(1,npadb))
        rbblat = lat[-1,0] + dyb.squeeze()[-1]*np.tile(np.arange(-npadb,0)[None,:],(npadr,1))
    else:
        rbblon, rbblat = np.zeros((npadr,npadb)), np.zeros((npadr,npadb))
    if npadr > 0 and npadt > 0:
        rtblon = lon[-1,-1] + dxr.squeeze()[-1]*np.tile(np.arange(1,npadr+1)[:,None],(1,npadt))
        rtblat = lat[-1,-1] + dyt.squeeze()[-1]*np.tile(np.arange(1,npadt+1)[None,:],(npadr,1))
    else:
        rtblon, rtblat = np.zeros((npadr,npadt)), np.zeros((npadr,npadt))
    if npadl > 0 and npadt > 0:
        ltblon = lon[0,-1] + dxl.squeeze()[-1]*np.tile(np.arange(-npadl,0)[:,None],(1,npadt))
        ltblat = lat[0,-1] + dyt.squeeze()[0]*np.tile(np.arange(1,npadt+1)[None,:],(npadl,1))
    else:
        ltblon, ltblat = np.zeros((npadl,npadt)), np.zeros((npadl,npadt))
        
    ### assemblate blocks to form output matrix
    lon_pad = np.bmat([[lbblon,lblon,ltblon],[bblon,lon,tblon],[rbblon,rblon,rtblon]]).A
    lat_pad = np.bmat([[lbblat,lblat,ltblat],[bblat,lat,tblat],[rbblat,rblat,rtblat]]).A     
    #print(lbblon.shape,lblon.shape,ltblon.shape)
    #print(bblon.shape,lon.shape,tblon.shape)
    #print(rbblon.shape,rblon.shape,rtblon.shape)
    
    return lon_pad, lat_pad
