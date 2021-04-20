##################################
# IMPORTS
##################################
import os
import sys
import numpy as np
import pandas as pd
import gootools as gt
from glob import glob
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
import astropy.constants as const
import gootools as gt
##################################
# CONSTANTS
##################################
pixels = u.def_unit('pixels', 12.27*u.mas)
iwa = 150/12.26 # inner working angle in mas
pupil_offset = 135.99 # pupil-offset (deg) in pupil-tracking mode (SPHERE User Manual P99.0, 6th public release, P99 Phase 1)
true_north_correction = -1.75 # true North correction (deg) (SPHERE User Manual P99.0, 6th public release, P99 Phase 1)
# file locations
datadir = "D:/Science/LBV/Data/images/"
figdir = "C:/Users/Keyan/Desktop/Science/Data/LBV Coronagraphy/Figures/raw/"
files = np.array(glob(datadir+"*"))
darks = np.array(glob(datadir+'*DARK*'))
flats = np.array(glob(datadir+'*FLAT*'))
centers = np.array(glob(datadir+'*CENTER*'))
fluxs = np.array(glob(datadir+'*FLUX*'))
sky = np.array(glob(datadir+'*SKY*'))
calibrations = np.r_[darks,flats,fluxs,sky,centers]
sat_files = ['D:/Science/LBV/Data/images\\AS-314-IRDIS_CI.SKY20.fits', 
             'D:/Science/LBV/Data/images\\HD-160529-IRDIS_CI.SKY190.fits', 
             'D:/Science/LBV/Data/images\\HD-316285-IRDIS_CI.SKY108.fits', 
             'D:/Science/LBV/Data/images\\HD-326823-IRDIS_CI.SKY79.fits', 
             'D:/Science/LBV/Data/images\\HD-326823-IRDIS_CI.SKY99.fits', 
             'D:/Science/LBV/Data/images\\MWC-314-IRDIS_CI.SKY263.fits', 
             'D:/Science/LBV/Data/images\\MWC-314-IRDIS_CI.SKY283.fits', 
             'D:/Science/LBV/Data/images\\zet01-Sco-IRDIS_CI.SKY286.fits', 
             'D:/Science/LBV/Data/images\\zet01-Sco-IRDIS_CI.SKY306.fits', 
             'D:/Science/LBV/Data/images\\zet01-Sco-IRDIS_CI.SKY337.fits']
targets = np.array(['AS-314',
                    'HD-160529', 
                    'HD-168607', 
                    'HD-168625',
                    'HD-316285', 
                    'HD-326823', 
                    'MWC-314',  
                    'ZETA-SCO'])
target_H_flux = {
     'AS 314'    : 7.63,
     'HD 160529' : 3.23,
     'HD 168607' : 3.88,
     'HD 168625' : 4.537,
     'HD 316285' : 4.227,
     'HD 326823' : 6.1,
     'MWC 314'   : 5.54,
     'Zeta01 Sco': 3.27
                }
# date of each observation in mjd
mjd = np.array([58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253., 58253.,
                58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 58281., 
                58368., 58368., 58356., 58356., 58368., 58356., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58368., 58356., 58356., 58356., 58368., 58368., 58368., 58368., 
                58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 58365., 
                58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 58275., 
                58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238., 58238.,
                58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 58269., 
                58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274., 58274.])
center_coordinates = np.array([ # x(left),y(left),x(right),y(right) for each object
    [477.13,522.33,1504.25,511.14],
    [477.04,517.34,1504.21,506.16],
    [477.90,524.85,1504.96,513.32],
    [476.43,525.09,1503.50,513.62],
    [475.54,521.75,1502.70,510.60],
    [477.56,525.98,1504.74,514.74],
    [476.24,521.26,1503.37,510.06],
    [476.29,521.93,1503.48,510.72]
])
satellite_spots = np.array([ # locations of satelite spots for each object, then each spot, then left and right x,y values
    [[[510.20267972,442.25140407, 441.82659351, 509.99480703] ,[513.39863627, 445.37420655, 444.90750795, 513.19440029]],[[555.04086046, 555.02946308, 487.38992481, 487.64666913], [544.01673722, 543.84016792, 476.07787127, 476.36600949]]],
    [[[510.661, 441.67794907, 441.5391109, 510.07437834] ,[513.83224358, 444.77958217, 444.65606972, 513.2679245 ]],[[550.55934245, 550.31963664, 482.23861263, 482.66973094], [539.42670261, 539.20678614, 470.96580194, 471.4742838 ]]],
    [[[511.29560983,442.60569192, 442.54033967, 511.11322478], [514.45139536, 445.58763468, 445.4900223, 514.2381271 ]],[[557.85070161, 557.78370378, 489.88339658, 490.00019435], [546.3999515, 546.28698997, 478.24915378, 478.44005542]]],
    [[[509.96812938,441.12397188, 441.07887294, 509.44256574] ,[512.95905052, 444.15467618, 443.99162404, 512.62805251]],[[558.19549675, 558.12510081, 490.16329544, 490.35480454], [546.69375484, 546.64699451, 478.507453, 478.80215929]]],
    [[[509.25837527,440.2114817 ,440.15192854 ,508.58368272] ,[512.45278237 ,443.36682199, 443.22685552, 511.79880508]],[[555.26106462, 554.7073956, 486.67487507, 487.07280009] ,[544.14970328, 543.54910045, 475.44590368, 475.87316522]]],
    [[[510.962382, 442.37596764, 442.17566763, 510.99625435] ,[514.39310207, 445.42419271, 445.26415195, 514.06896393]],[[559.11301375, 558.87897859, 490.86997577, 490.81790903], [548.12778869, 547.74685987, 479.55520543, 479.71221485]]],
    [[[509.64866813,440.99883897, 440.92251301, 509.25622124] ,[513.00802965, 444.22216801, 443.92975182, 512.47387561]],[[554.78633448, 554.6222336, 486.61742162, 486.96082602] ,[543.87393724, 543.31706125, 475.26180191, 475.67249164]]],
    [[[509.75756024,440.7074172 ,440.94019747 ,509.45581237], [512.92700603 ,443.8999537, 444.03626723, 512.67938684]],[[555.08034028, 554.96100084, 486.89433132, 487.32482143], [543.85427975, 543.71765984, 475.589786, 476.08453377]]]
])
axis_center = {
    'AS-314':(477.13,522.33),
    'HD-160529':(477.04,517.34), 
    'HD-168607':(477.90,524.85), 
    'HD-168625':(476.43,525.09),
    'HD-316285':(475.54,521.75), 
    'HD-326823':(477.56,525.98), 
    'MWC-314':(476.24,521.26),  
    'ZETA-SCO':(476.29,521.93)
}
##################################
# FUNCTIONS
##################################
# UNITS
def pix2au(pix,distance):
    """
    Convert pixel values to projected distances in astronomical units
    """
    return (pix*pixels).to('radian').value*distance*u.kpc.to('AU')
def au2pix(au,distance):
    """
    Convert projected distances in astronomical units, to pixel angular seperations
    """
    return ((au*u.AU).to('kpc')/(distance*u.kpc))*u.radian.to(pixels)

# PLOTTING
def centered_crop(image,radius):
    """
    better for contrast
    """
    side_length = int(np.floor(image.shape[0]/2))
    limits = [int(np.floor(side_length-radius)),int(np.floor(side_length+radius))]
    return image[limits[0]:limits[1],limits[0]:limits[1]]
def mask_coronagraph(ax):
    """
    Place a black circle over the inner working angle of an image
    """
    half = (ax.get_xlim()[1]+.5)/2
    ax.add_patch(Circle((half,half),radius=150/12.26,color='black'))
def lbv_show(fig,ax,image,name,cmap='viridis',full_range=False,**kwargs):
    """
    Nice way to format images of these lbvs
    """
    plate_scale = 12.26
    arcsec = 1000/plate_scale
    radii = (np.arange(10)+1)*arcsec/2
    gt.logshow(fig,ax,image,cmap=cmap,full_range=False,**kwargs)
    mask_coronagraph(ax)
    center = ((ax.get_xlim()[1]+.5)/2,(ax.get_xlim()[1]+.5)/2)
    for i,r in enumerate(radii):
        ax.add_patch(Circle(center,radius=r,fill=False,linestyle='--',edgecolor='white',linewidth=.5,alpha=.25,figure=fig))
        if i%2==1:
            ax.text(*(center+np.array([-1,r+2])),'{0:.1f}``'.format((i+1)/2),fontsize=8,ha='right',color='white').set_clip_on(True)
            ax.text(*(center+np.array([5,r+2])),'{0:.1f}kAU'.format(pix2au(r,LBVs().get_distance(name))/1000),fontsize=8,ha='left',color='white').set_clip_on(True)
    ax.axvline(center[0],linestyle='--',alpha=.25,color='white',linewidth=.5)
    ax.axhline(center[0],linestyle='--',alpha=.25,color='white',linewidth=.5)
#     ax.set_title(name,fontsize=11)
    ax.grid()  
def center_lim(ax,radius):
    """
    Set the axis limits to be a square of side length 2*radius, centered on our object
    """
    ax.set_xlim(1024/2-radius,1024/2+radius)
    ax.set_ylim(1024/2-radius,1024/2+radius)

# DATA EXTRACTION
def load(file,**kwargs):
    data,header = fits.getdata(file),fits.getheader(file)
    return data,header
def header(file,**kwargs):
    header=fits.getheader(file)
    return header
def data(file,**kwargs):
    data = fits.getdata(file)
    return data
def load_preprocessed(name):
    im = 'data/{}.txt'.format(name)
    err = 'data/{}_error.txt'.format(name)
    return np.loadtxt(im,delimiter=','),np.loadtxt(err,delimiter=',')
def load_irdap_flux_cal(name):
    file = "D:/Science/LBV/test_irdap/" + name + '/calibration/flux/' + name.replace('-','_') + '_2018-05-15_reference_flux.csv'
    return pd.read_csv(file,sep=',')

# PROFILES
def cut_line(x,theta,xshift=0,yshift=0):
    """
    theta is a position angle
    """
    slope = np.tan(np.deg2rad(-1*(theta-90)))
    y = slope*(x-512-xshift)+512+yshift
    return y
def signed_distance(x,y,theta):
    """
    Negative values are to the left, positive to the right of center
    """
    sign = 1
    if x<0:
        sign = -1
    elif theta==0 and y>0:
        sign = -1
    return sign*np.hypot(x,y)
def cut(image,error,theta,xshift=0,yshift=0):
    
    sign = 1
    if (theta > 90 or theta==0) and theta!=180:
        sign = -1
        
    l,flux,err = list(),list(),list()
    f = lambda x : cut_line(x,theta,xshift=xshift,yshift=yshift)
    
    if theta==0:
        x,y = 512,1000
    elif theta==180:
        x,y = 512,0
    else:
        x,y = 0,0
        while f(x) < 0 or f(x) > 1000:
            x+=1
        y = int(np.floor(f(x)))
    while x<1000 and (y<=1000 and y>=0):
        l.append(signed_distance(x-512-xshift,y-512-yshift,theta))
        flux.append(image[x,y])
        err.append(error[x,y])
        while (sign*(y+sign - f(x+1)) <= 0 or theta in [0,180]) and (y<=1000 and y>=0):
            y+=sign
            l.append(signed_distance(x-512-xshift,y-512-yshift,theta))
            flux.append(image[x,y])
            err.append(error[x,y])
        x+=1
    mask = np.where(np.abs(l)>iwa)
    return np.array(l)[mask],np.array(flux)[mask],np.array(err)[mask]
def cut_image(image,error,theta,name,xshift=0,yshift=0,lim=(0,1000)):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=[20,10])
    l,flux,err = cut(image,error,theta,xshift=xshift,yshift=yshift)
    coronagraph_mask = np.where((l<-1*iwa)|(l>iwa))
    l,flux,err = l[coronagraph_mask],flux[coronagraph_mask],err
    
    lbv_show(fig,ax1,image,name)
    ax1.plot((x:=np.linspace(0,1024)),cut_line(x,theta))
    ax1.set_xlim(*lim)
    ax1.set_ylim(*lim)
    
    mask = np.where((np.abs(l)>iwa)&(np.abs(l)<400))
    ax2.fill_between(l[mask],np.log10(flux[mask]+err[mask]),np.log10(flux[mask]-err[mask]),alpha=.3)
    ax2.plot(l[mask],np.log10(flux)[mask])
    ax2.axvspan(-1*iwa,iwa,color='black',zorder=1000)
    ax2.set_ylim(0,np.log10(flux)[mask].max())
    ax2.set_xlim(-400,400)
    
##################################
# CLASSES
##################################
class LBVs():
    def __init__(self):
        self.names = np.array(["AS-314","HD-160529","HD-168607","HD-168625","HD-316285","HD-326823","MWC-314","ZETA-SCO"])
        self.logT = np.array([4.01,[4,3.9],3.97,4.15,4.18,4.29,4.48,4.29],dtype=object)
        self.logL = np.array([4.9,5.46,5.38,5.23,5.48,4.98,6.34,6.02])
        self.distance = np.array([1.5,2.1,1.46,1.51,4.56,1.27,3.93,2.52])
        self.radius = np.array([89.671237, [178.91764, 283.56535], 187.34977, 68.810254, 79.919574, 27.080281, 54.032263, 89.671237],dtype=object)
        self.sdor_type = np.array(['pf+','s-a','w-a','ex','ex','ex','pf+','ex'])
        self.confirmed = np.array([False,True,True,False,False,False,False,False])
        self.is_binary = np.array([False,False,False,True,False,True,True,False])
    
    def __repr__(self):
        return repr(pd.DataFrame({"Star":self.names,
                                  "logT":self.logT,
                                  "logL":self.logL,
                                  "Radius [$R_\sun$]":self.radius,
                                  "Distance [kpc]":self.distance,
                                  "S Dor Type":self.sdor_type,
                                  "Confirmed LBV?":self.confirmed,
                                  "Binary?":self.is_binary}))
    
    def get_distance(self,name):
        return self.distance[np.where(self.names==name.upper())][0]
    
    def hrd(self,save=False):
        if save:
            fig,ax=plt.subplots(1,figsize=(4,4))
            #Making the instability strips
            ax.axvspan(np.log10(8000)-.05,np.log10(8000)+.05,alpha=0.5)
            t = [3.953,3.953,4.5,4.35]
            l = [5.45,5.05,6.5,6.5]
            ax.add_patch(patch.Polygon(xy=list(zip(t,l)),alpha=.5)) 
            ax.set_xlim(4.5,3.75)
            ax.set_ylim(4.8,6.5)
            ax.set_xlabel("log$T_{eff}$",fontsize=14)
            ax.set_ylabel(r"log$\frac{L}{L_{sun}}$",fontsize=14)
            #Plotting points
            for t,l,n in zip(self.logT,self.logL,self.names):
                if type(t)==list:
                    ax.plot(t,[l,l],marker='s',markersize=10,color='black',zorder=100)
                    ax.text(t[1]-0.02,l-0.02,n,fontsize=10,zorder=100)
                else:
                    ax.scatter(t,l,marker='s',s=100,color='black',zorder=100)
                    ax.text(t-0.02,l-0.02,n,fontsize=10,zorder=100)
            plt.savefig('figures/hrd.jpeg',dpi=300,overwrite=True)
        else:
            fig,ax=plt.subplots(1,figsize=(15,15))
            #Making the instability strips
            ax.axvline(np.log10(8000),linestyle='--',lw=5,alpha=.5)
            t = [3.953,3.953,4.5,4.35]
            l = [5.45,5.05,6.5,6.5]
            ax.add_patch(patch.Polygon(xy=list(zip(t,l)),alpha=.5)) 
            ax.set_xlim(4.5,3.75)
            ax.set_ylim(4.8,6.5)
            ax.set_xlabel("log$T_{eff}$",fontsize=20)
            ax.set_ylabel(r"log$\frac{L}{L_{sun}}$",fontsize=20)
            #Plotting points
            for t,l,n in zip(self.logT,self.logL,self.names):
                if type(t)==list:
                    ax.plot(t,[l,l],marker='s',markersize=10,color='black',zorder=100)
                    ax.text(t[1]-0.02,l-0.02,n,fontsize=18,zorder=100)
                else:
                    ax.scatter(t,l,marker='s',s=100,color='black',zorder=100)
                    ax.text(t-0.02,l-0.02,n,fontsize=18,zorder=100)
            plt.show()