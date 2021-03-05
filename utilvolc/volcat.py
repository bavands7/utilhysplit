# volcat.py
# A reader for VOLCAT data using xarray
# For use with MONET
import sys
import os
from os import walk
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import numpy as np
import numpy.ma as ma
import pandas as pd

"""
This script contains routines that open/read VOLCAT data in xarray format, 
manipulate arrays as necessary, and plots desirable variables.
-------------
Functions:
-------------
open_dataset: opens single VOLCAT file
open_mfdataset: opens multiple VOLCAT files
bbox: finds bounding box around data - used for trimming arrays
_get_time: set time dimension for VOLCAT data
_get_latlon: rename lat/lon, set coordinates of VOLCAT data
get_height: returns array of ash top height from VOLCAT
get_radius: returns array of ash effective radius from VOLCAT
get_mass:  returns array of ash mass loading from VOLCAT
get_atherr: returns array of ash top height error from VOLCAT
plot_height: plots ash top height from VOLCAT
plot_radius: plots ash effective radius from VOLCAT
plot_mass: plots ash mass loading from VOLCAT
------------
"""


def open_dataset(fname, pc_correct=False):
    """Opens single VOLCAT file"""
    dset = xr.open_dataset(fname, mask_and_scale=False, decode_times=False)
    # not needed for new Bezy data.
    try:
        dset = dset.rename({"Dim1": 'y', "Dim0": 'x'})
    except:
        pass
    # use parallax corrected if available and flag is set.
    if 'pc_latitude' in dset.data_vars and pc_correct:
        # rename uncorrected values
        dset = \
             dset.rename({'latitude':'uc_latitude','longitude':'uc_longitude'})
        # rename corrected values
        dset = \
             dset.rename({'pc_latitude':'latitude','pc_longitude':'longitude'})
        dset.attrs.update({'parallax corrected coordinates':'True'})  
    else:  
        dset.attrs.update({'parallax corrected coordinates':'False'})  
    dset = _get_latlon(dset,'latitude','longitude')
    dset = _get_time(dset)
    return dset

def find_volcat(tdir, vid=None, daterange=None, verbose=False):
    """
    tdir : str
    daterange : [datetime, datetime] or None
    Locates files in tdir which follow the volcat naming
    convention as defined in VolcatName class.
    If a daterange is defined will return only files 

    Returns:
    vnlist : dictionary with key=date and value=filename.

    """
    vnhash = {}
    nflist = []
    if not os.path.isdir(tdir):
         print('directory not valid {}'.format(tdir))
    for (dirpath,dirnames,filenames) in walk(tdir):
         for fln in filenames:
             try:
                vn = VolcatName(fln)
             except:
                if verbose: print('Not VOLCAT filename {}'.format(fln))
                nflist.append(fln)
                continue
             if daterange:
                if vn.date < daterange[0] or vn.date > daterange[1]:
                   continue
             if vid and vn.vhash['volcano id'] != vid: continue
             if vn.date not in vnhash.keys():
                vnhash[vn.date] = vn
             else: 
                print('two files with same date')
                print(vnhash[vn.date].compare(vn))
                
    return vnhash

def test_volcat(tdir, daterange=None, verbose=True):
    """
    checks the pc_latitude field for values greater than 0.
    """
    vnlist = find_volcat(tdir, daterange, verbose)
    for key in vnlist.keys():
        vname = vnlist[key].fname
        dset = open_dataset(os.path.join(tdir,vname),pc_correct=False)
        if np.max(dset.pc_latitude) > 0: 
           print('passed')
        else:
           print('failed')
         

class VolcatName:
    """
    12/18/2020 works with 'new' data format.
    parse the volcat name to get information.
    attributes:
    self.fname name of file
    self.date date associated with file
    self.vhash is a dictionary which contains info
    gleaned from the naming convention.

    methods:
    compare: returns what is different between two file names.
    """
    def __init__(self,fname):
        self.fname=fname 
        self.vhash = {} 
        self.date = None
        self.dtfmt = "s%Y%j_%H%M%S"

        self.keylist = ['algorithm name']
        self.keylist.append('satellite platform')
        self.keylist.append('event scanning strategy') 
        self.keylist.append('event date') 
        self.keylist.append('event time') 
        self.keylist.append('volcano id') 
        self.keylist.append('description') 
        self.keylist.append('WMO satellite id') 
        self.keylist.append('image scanning strategy') 
        self.keylist.append('image date') 
        self.keylist.append('image time') 
        self.keylist.append('feature id') 

        self.parse(fname)

    def compare(self, other):
        """
        other is another VolcatName object.
        Returns
        dictionary of information which is different.
        values is a  tuple of (other value, self value).
        """ 
        diffhash = {}
        for key in self.keylist:
            if other.vhash[key] != self.vhash[key]:
                diffhash[key] = (other.vhash[key], self.vhash[key])
        return diffhash

    def __str__(self):
        val = [self.vhash[x] for x in self.keylist]
        return str.join('_',val)

    def parse(self,fname):
        temp = fname.split('_')
        for val in zip(self.keylist,temp):
            self.vhash[val[0]] = val[1]
        # use event date?
        dstr = '{}_{}'.format(self.vhash[self.keylist[3]],
                              self.vhash[self.keylist[4]])
        self.date = datetime.datetime.strptime(dstr,self.dtfmt)
        self.vhash[self.keylist[11]] = \
            self.vhash[self.keylist[11]].replace('.nc','')
        return self.vhash

    def create_name(self):
        """
        To do: returns filename given some inputs.
        """
        return -1


def open_dataset2(fname):
    """Opens single VOLCAT file in reventador format """
    print(fname)
    dset = xr.open_dataset(fname, mask_and_scale=False, decode_times=False)
    #dset = dset.rename({"Dim1":'y',"Dim0":'x'})
    #dset = _get_latlon(dset)
    #dset = _get_time(dset)
    return dset


def open_mfdataset(fname):
    # 12/1/2020 Not modified for new files (Bezy) 
    """Opens multiple VOLCAT files"""
    print(fname)
    dset = xr.open_mfdataset(fname, concat_dim='time', decode_times=False, mask_and_scale=False)
    from glob import glob
    from numpy import sort
    files = sort(glob(fname))
    das = []
    for i in files:
        das.append(open_dataset(i))
    dset = xr.concat(das, dim='time')
    dset = _get_latlon(dset)
    dset = dset.rename({"lines": 'y', "elements": 'x'})
    return dset


def bbox(darray):
    """Returns bounding box around data
    Input: Must be dataarray
    Outupt: Lower left corner, upper right corner of bounding box
    around data"""
    import numpy as np
    arr = darray[0, :, :].values
    a = np.where(arr != -999.)
    if np.min(a[0]) != 0. and np.min(a[1]) != 0.:
        bbox = ([np.min(a[0]-3), np.min(a[1])-3], [np.max(a[0]+3), np.max(a[1])+3])
    else:
        bbox = ([np.min(a[0]), np.min(a[1])], [np.max(a[0]), np.max(a[1])])
    return bbox

def _get_latlon(dset,name1='latitude',name2='longitude'):
    dset = dset.set_coords([name1, name2])
    return dset

def _get_time(dset):
    import pandas as pd
    temp = dset.attrs['time_coverage_start']
    time = pd.to_datetime(temp)
    dset['time'] = time
    dset = dset.expand_dims(dim='time')
    dset = dset.set_coords(['time'])
    return dset

# Extracting variables
def get_data(dset,vname,clip=True):
    gen = dset.data_vars[vname]
    if clip:
        box = bbox(gen)
        gen = gen[:, box[0][0]:box[1][0], box[0][1]:box[1][1]]
        gen = gen.where(gen != gen._FillValue)
    return gen

def check_names(dset,vname,checklist,clip=True):
    if vname:
        return get_data(dset,vname,clip=clip)
    for val in checklist:
        if val in dset.data_vars:
           return get_data(dset,val,clip=clip)
    return xr.DataArray()


def create_pc_plot(dset):
    """
    creates plots of parallax corrected vs. uncorrected values.
    """   
   
    def subfunc(ax, vals):
        ax.plot(vals[0], vals[1], 'k.', MarkerSize=1) 
        # plot 1:1 line
        minval = np.min(vals[0])
        maxval = np.max(vals[0])
        ax.plot([minval,maxval],[minval,maxval],'--r')

    latitude, longitude = compare_pc(dset)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.set_ylabel('uncorrected')
    ax2.set_ylabel('uncorrected')
    ax2.set_xlabel('corrected')

    subfunc(ax1, latitude) 
    subfunc(ax2, longitude) 
    return fig, ax1, ax2

def compare_pc(dset):
    """
    Returns:
    latitude : [list of parrallax corrected values, list of uncorrected values]
    longitude : [list of parrallax corrected values, list of uncorrected values]
    """
    def process(pc,val):
        # pair corrected and uncorrected values.
        pzip = list(zip(pc,val))
        # remove nans
        new = [x for x in pzip if not np.isnan(x[0])]
        return list(zip(*new))  

    pc_lat = get_pc_latitude(dset)
    pc_lon = get_pc_longitude(dset)
    latvals = pc_lat.latitude.values.flatten()
    lonvals = pc_lon.longitude.values.flatten()
    pclat = pc_lat.values.flatten()
    pclon = pc_lon.values.flatten()

    latitude = process(pclat, latvals)
    longitude = process(pclon,lonvals)
    return latitude, longitude 


def get_pc_latitude(dset,vname=None,clip=True):
    """Returns array with retrieved height of the highest layer of ash."""
    """Default units are km above sea-level"""
    checklist = ['pc_latitude']
    return check_names(dset,vname,checklist,clip=clip)

def get_pc_longitude(dset,vname=None,clip=True):
    """Returns array with retrieved height of the highest layer of ash."""
    """Default units are km above sea-level"""
    checklist = ['pc_longitude']
    return check_names(dset,vname,checklist,clip=clip)

def get_height(dset,vname=None,clip=True):
    """Returns array with retrieved height of the highest layer of ash."""
    """Default units are km above sea-level"""
    checklist = ['ash_cth','ash_cloud_height']
    return check_names(dset,vname,checklist,clip=clip)

def get_radius(dset,vname=None,clip=True):
    """Returns 2d array of ash effective radius"""
    """Default units are micrometer"""
    checklist = ['ash_r_eff','effective_radius_of_ash']
    return check_names(dset,vname,checklist,clip=clip)

def get_mass(dset,vname=None,clip=True):
    """Returns 2d array of ash mass loading"""
    """Default units are grams / meter^2"""
    checklist = ['ash_mass','ash_mass_loading']
    return check_names(dset,vname,checklist,clip=clip)

def mass_sum(dset):
    mass = get_mass(dset)
    mass2 = mass.where(mass > 0., 0.0).values
    mass_sum = np.sum(mass2)
    return mass_sum

def get_time(dset):
    time = dset.time_coverage_start
    return time

def get_atherr(dset):
    """Returns array with uncertainty in ash top height from VOLCAT."""
    """Default units are km above sea-level"""
    height_err = dset.ash_cth_uncertainty
    height_err = height_err.where(height_err != height_err._FillValue, drop=True)
    return height_err

def trim_arrray(dset):
    """Trim the VOLCAT array around data
    Make smaller for comparison to HYSPLIT """

# Plotting variables
def plot_height(dset):
    """Plots ash top height from VOLCAT
    Does not save figure - quick image creation"""
    fig = plt.figure('Ash_Top_Height')
    title = 'Ash Top Height (km)'
    ax = fig.add_subplot(1, 1, 1)
    plot_gen(dset, ax, val='height', time=None, plotmap=True,
             title=title)

def plot_radius(dset):
    """Plots ash effective radius from VOLCAT
    Does not save figure - quick image creation"""
    fig = plt.figure('Ash_Effective_Radius')
    title = 'Ash effective radius ($\mu$m)'
    ax = fig.add_subplot(1, 1, 1)
    plot_gen(dset, ax, val='radius', time=None, plotmap=True,
             title=title)

def plot_mass(dset):
    fig = plt.figure('Ash_Mass_Loading')
    ax = fig.add_subplot(1, 1, 1)
    plot_gen(dset, ax, val='mass', time=None, plotmap=True,
             title='Ash_Mass_Loading')


def plot_gen(dset, ax,  val='mass', time=None, plotmap=True,
              title=None):
      """Plot ash mass loading from VOLCAT
      Does not save figure - quick image creation"""
      #lat=dset.latitude
      #lon=dset.longitude
      if val == 'mass':
          mass=get_mass(dset)
      elif val == 'radius':
          mass=get_radius(dset)
      elif val == 'height':
          mass=get_height(dset)
      if time and 'time' in mass.coords:
         mass = mass.sel(time=time)
      elif 'time' in mass.coords:
         mass = mass.isel(time=0)
      lat=mass.latitude
      lon=mass.longitude
      if plotmap:
          m=plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
          m.add_feature(cfeat.LAND)
          m.add_feature(cfeat.COASTLINE)
          m.add_feature(cfeat.BORDERS)
          plt.pcolormesh( lon, lat, mass, transform=ccrs.PlateCarree())
      else:
          plt.pcolormesh( lon, lat, mass )
      plt.colorbar()
      plt.title(title)
      plt.show()


def correct_pc(dset):
    mass = get_mass(dset,clip=False)
    height = get_height(dset,clip=False)
    newmass = xr.zeros_like(mass.isel(time=0))
    newhgt = xr.zeros_like(height.isel(time=0))
    time = mass.time 
    pclat = get_pc_latitude(dset,clip=False)
    pclon = get_pc_longitude(dset,clip=False)


    fill = mass.attrs['_FillValue']
    vpi = np.where(mass != fill)
    print(fill)
    print(mass.shape)
    print(vpi)
    pclat = pclat[vpi].values.flatten()
    pclon = pclon[vpi].values.flatten()
    mlist = mass[vpi].values.flatten()
    hlist = height[vpi].values.flatten()       
    pclat = [x for x in pclat if not np.isnan(x)]
    pclon = [x for x in pclon if not np.isnan(x)]
    mlist =  [x for x in mlist if not np.isnan(x)]
    hlist =  [x for x in hlist if not np.isnan(x)]
 
    #pclat = [x for x in pclat.values.flatten() if not np.isnan(x)]
    #pclon = [x for x in pclon.values.flatten() if not np.isnan(x)]
    #mlist =  [x for x in mass.values.flatten() if not np.isnan(x)]
    #hlist =  [x for x in height.values.flatten() if not np.isnan(x)]

    indexlist = []
    for point in zip(pclon, pclat,mlist,hlist):
        iii = mass.monet.nearest_ij(lat=point[1], lon=point[0])
        print(iii, point[2])
        newmass = xr.where((newmass.coords['x']==iii[0]) & (newmass.coords['y']==iii[1]), 
                            point[2], newmass)         
        newhgt = xr.where((newhgt.coords['x']==iii[0]) & (newhgt.coords['y']==iii[1]), 
                            point[3], newhgt)         
        # keeps track of new indices of lat lon points.
        indexlist.append(iii) 
    # check if any points are mapped to the same point.
    if len(indexlist) != len(list(set(indexlist))):
       print('WARNING: correct_pc function: some values mapped to same point')
    
    newmass = newmass.assign_attrs({'_FillValue':0})
    newhgt = newhgt.assign_attrs({'_FillValue':0})

    newmass = newmass.expand_dims("time")
    newhgt = newhgt.expand_dims("time")
    #newmass.assign_coords({"time":time})

    dnew = xr.Dataset({'ash_mass_loading':newmass,'ash_cloud_height':newhgt})
    #dnew['ash_mass_loading'].assign_attrs('_FillValue') = 0
    #dnew['ash_cloud_height'].assign_attrs('_FillValue') = 0
    return dnew     





