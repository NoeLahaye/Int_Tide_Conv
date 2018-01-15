# Density of seawater at atmospheric pressure (0.1 MPa) using Eq. (8)
#   given by [1] which best fit the data of [2] and [3]. The pure water
#   density equation is a best fit to the data of [4]. 
#   Values at temperature higher than the normal boiling temperature are
#   calculated at the saturation pressure.
#
# INPUT:  (all must have same dimensions)
#   T = temperature [degree C] (ITS-90)
#   S = salinity    [g/kg] (reference-composition salinity)
#
# OUTPUT:
#   rho = density   [kg/m^3]
#
# AUTHOR:  
#   Mostafa H. Sharqawy 12-18-2009, MIT (mhamed@mit.edu)
#
# DISCLAIMER:
#   This software is provided "as is" without warranty of any kind.
#   See the file sw_copy.m for conditions of use and licence.
# 
# VALIDITY: 0 < T < 180 C; 0 < S < 160 g/kg;
# 
# ACCURACY: 0.1#
# 
# REFERENCES:
#   [1] Sharqawy M. H., Lienhard J. H., and Zubair, S. M., Desalination and Water Treatment, 2009.
#   [2] Isdale, and Morris, Desalination, 10(4), 329, 1972.
#   [3] Millero and Poisson, Deep-Sea Research, 28A (6), 625, 1981
#   [4] IAPWS release on the Thermodynamic properties of ordinary water substance, 1996. 


def SW_Density(T,S): 
    if S.shape != T.shape: 
        exit('T and S must have same dimensions')
 
    S=S/1000. 
    a1=9.9992293295e+02;a2=2.0341179217e-02;a3=-6.1624591598e-03;a4=2.2614664708e-05;a5=-4.6570659168e-08;
    b1=8.0200240891e+02;b2=-2.0005183488e+00;b3=1.6771024982e-02;b4=-3.0600536746e-05;b5=-1.6132224742e-05;
    rho_w = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4
    D_rho = b1*S + b2*S*T + b3*S*T**2 + b4*S*T**3 + b5*S**2*T**2
    rho   = rho_w + D_rho
    
    return rho
