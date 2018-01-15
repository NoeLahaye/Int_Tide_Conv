from numpy import dot,meshgrid,linspace
from numpy import c_ as concat
from numpy.linalg import inv
def detrend_2d(data):
    ''' compute best bilinear fit (solving least square problem) and return detrended data'''
    [l,m] = data.shape

    x = linspace(-1,1,m)
    y = linspace(-1,1,m)
    [xx,yy] = meshgrid(x,y)
    X = concat[xx.ravel(), yy.ravel()]

    iX = inv(dot(X.T,X))

    coef = dot(iX,dot(X.T,data.ravel()))
                    
    return data - coef[0]*xx - coef[1]*yy - data.mean()
