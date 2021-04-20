# Problems while running [IRDAP](www.github.com/robvanholstein/IRDAP)
---
##### 1. our calibration files have 'ESO DPR CATG' keyword as ACQUISITION instead of CALIB and the filter as FILT_BBF_ND_H instead of FILT_BBF_H. Also our files have a couple weird header kewords so we just have to delete those (they have to do with the pressure inside of the detectors?).

    Solution: try changing that keyword to CALIB
~~~python
for f in glob(irdap_test_path+'/raw/*'): 
    fix = False
    with fits.open(f,output_verify='ignore') as hdul:
        if hdul[0].header['ESO DPR CATG'] == 'ACQUISITION':
            data = hdul[0].data.copy()
            new_header = hdul[0].header.copy()
            new_header['ESO DPR CATG'] = 'CALIB'
            new_header['ESO INS1 FILT ID'] = 'FILT_BBF_H'
            del new_header['ESO INS1 SENS101 VAL']
            del new_header['ESO INS4 SENS424 VAL']
            fix=True
    if fix:
        fits.writeto(f,data,header=new_header,overwrite=True)
~~~
---
##### 2. OSError: One or more files of the on-sky data do not have the NIR half-wave plate inserted.
I think this is just irdap picking up on how this isn't polarimetry?

    Solution: commented out lines between 600 and 1120 in irdap.py having to do with polarimetry
---
##### 3. IRDAP wants the different OBJECT, SKY, and CENTER files (at least) to be of the same exposure time.

    Solution: when saving the files make sure all of these files are normalized and the exposure time is changed to 1 second in the header

---
##### 4. IRDAP wants the same number of flats and darks

    Solution: eh?? 
---
##### 5. IRDAP wants multiple flats/darks

    Solution: probably best to just find more flats, that way I won't have to monkey around too much
---
##### 6. Need matching neutral density filters
    
    Solution: this problem seems pretty legit, so just take the object files that have 
~~~python 
header['ESO INS4 FILT2 NAME']='OPEN'
~~~