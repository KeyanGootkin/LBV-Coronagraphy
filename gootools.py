import sys
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import kplot.kg

def erase_print(text):
    """
    Takes the input text, and prints it after erasing the most recently printed line.
    """
    sys.stdout.write('\r'+text)
    
def progress_bar(n,N,width=100):
    """
    Insert into loops to track what fraction (n/N) of the way through the loop you are.
    """
    r = np.floor_divide(n*width,N)
    bar = "|"+"█"*r+" "*(width-r)+"|"
    per = '{0:.1f}%'.format((n/N)*100)
    erase_print(bar+per)
    
def chime(T=2,f=350):
    """
    Makes a beep of frequence f [Hz] which lasts for T [seconds].
    """
    t = np.arange(T*10000)/10000
    octave_mask=np.where(t<.1*T)
    ot = t.copy()
    ot[octave_mask]=0
    fifth_mask =np.where(t<.4*T)
    ft = t.copy()
    ft[fifth_mask]=0
    wave = (1-(t/T)**2)*(np.sin(2*np.pi*f*t) + np.sin(2*np.pi*f*1.5*ot) + np.sin(2*np.pi*f*4/3*ft))
    
    return Audio(wave,rate=10000,autoplay=True)

def logshow(fig,ax,image,cmap='viridis',full_range=False,**kwargs):
    if full_range:
        vmin=image.min()
    else:
        vmin=0
    norm = matplotlib.colors.SymLogNorm(linthresh=10.0, linscale=.10, vmin=vmin,vmax=image.max(),base=10)
    img = ax.imshow(image, origin='lower', cmap=cmap,norm=norm,**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')