from astroquery.eso import Eso
from pyvo.dal import tap
import urllib
from astropy.io import fits
import os
import shutil
from glob import glob
import numpy as np
import sys

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


alldir='D:/Science/LBV/Data/new_download/'
datadir = input("Hello my friend! Where do you want me to put all them there files?? (ie. C:/Users/Keyan/Desktop/)")
if datadir[-1]!='/':
    raise ValueError("Remember to put a / at the end of the directory name please !!")
    
ESO_TAP_OBS = "http://archive.eso.org/tap_obs"
tapobs = tap.TAPService(ESO_TAP_OBS)

try:
    N=int(input("Hi Jamie! How many files do you want me to grab? (out of 355)\n"))
except:
    raise TypeError("Sorry friend, I don't think you gave me an integer :(")
    
query="""SELECT TOP %d ob_name, dp_type, mjd_obs, dp_id 
from dbo.raw
where prog_id = '0101.D-0064(A)'""" % (N)
res = tapobs.search(query=query)

nfiles = 0
progress_bar(0,1,width=100)
for row in res:
    nfiles += 1
    destination= datadir+row["ob_name"].decode()+'.'+row['dp_type'].decode()+str(nfiles)+'.Z'
    download_url = "http://archive.eso.org/datalink/links?ID=ivo://eso.org/ID?%s&eso_download=file" % row["dp_id"].decode()
    urllib.request.urlretrieve(download_url,filename=destination)
    progress_bar(nfiles,len(res),width=100)