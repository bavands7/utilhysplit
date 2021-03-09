# write_emitimes.py
# Writes a HYSPLIT EMITIMES.txt file for a cylindrical volcanic source
# and for inserting volcat data into hysplit
from utilhysplit import cylsource
from utilhysplit import emitimes
from datetime import datetime
from glob import glob
import xarray as xr
import numpy as np
import numpy.ma as ma
from utilvolc import volcat
from os import path
from math import pi, cos

"""
This script contains functions (class) that writes emitimes files
for cylinder source and volcat data insertion hysplit runs.
--------------
Functions:
--------------
write_cyl_file: writes emitimes file to working directory
Class: InsertVolcat

--------------
"""


def write_cyl_file(wdir, date_time, lat, lon, volcname, radius, dr, duration, pollnum, pollpercents, altitude, umbrella):
    """ Write EMITIMES.txt file for cylinder source hysplit runs
    Inputs:
    wdir: working directory (string)
    date_time: date and time of emission start (datetime object)
    lat: latitude for center of cylinder (float)
    lon: longitude for center of cylinder (float)
    volcname: name of volcano (string)
    radius: radius of cylinder around volcano vent in meters (integer)
    dr: will determine number of concentric circles in column,
    must be evenly divisible into radius (integer)
    duration: hour and minute (HHMM) of emission (string)
    pollnum: number of particle size bins (integer)
    pollpercents: percentage of particles for each size bin,
    represented as values from [0] to [1] (list)
    altitude: height of column in meters [bottom, top] (list)
    umbrella: 1 - uniform column (integer)

    Outputs:
    filename: location of newly written emitimes file (string)
    """

    dt = date_time
    fname = 'EMIT_'+volcname+'_cyl_'+dt.strftime("%Y%m%d%H%M")+'_'+duration+'hrs.txt'

    filename = wdir+fname

    # Creating emitimes files
    efile = cylsource.EmitCyl(filename=filename)
    latlist, lonlist = efile.calc_cyl(lat, lon, radius, dr)
    nrecords = efile.calc_nrecs(latlist, pollnum=pollnum, umbrella=umbrella)
    efile.write_data(dt, latlist, lonlist, nrecords, pollnum, duration=duration,
                     pollpercents=pollpercents, height=altitude)

    print('File '+filename+' created')

    return filename


class InsertVolcat:

    def __init__(self, wdir, vdir, date_time,
                 duration='0010',
                 pollpercents=1,
                 pollnum=[1],
                 vname=None,
                 vid=None):
        """
        Class of tools for inserting volcat data into hysplit
        -------------
        Inputs:
        wdir: working directory - where emitimes file will be located (string)
        vdir: volcat directory - where volcat data files are located (string)
        date_time: date and time of volcat file to use (datetime object)
        duration: hour and minute (HHMM) of emission - default is '0010' (string)
        pollnum: number of particle size bins - default is 1 (integer)
        pollpercents: percentage of particles for each size bin,
        represented as values from [0] to [1] - default is [1] (list)
        vname: volcano name - default is None (string)
        vid: volcano id - default is None (string)
        Outputs:
        --------------
        Functions:
        add_vname: adds volcano name
        find_match: finds string with date_time and vid to match
        find_fname: finds volcat filename
        get_area: calculates the domain area, gridded

        """

        if wdir[-1] != "/":
            wdir += "/"
        self.wdir = wdir
        if vdir[-1] != "/":
            vdir += "/"
        self.vdir = vdir
        self.date_time = date_time
        self.pollnum = pollnum
        self.pollpercents = pollpercents
        self.duration = duration
        self.vname = vname
        self.vid = vid
        self.find_fname()

    def add_vname(self, vname):
        """ Adds volcano name"""
        self.vname = vname

    def find_match(self):
        """Determines matching string to identify file"""
        if self.vid != None:
            match = self.date_time.strftime('%Y%j_%H%M%S_v')+self.vid
        else:
            match = self.date_time.strftime('%Y%j_%H%M%S')
        return match

    def find_fname(self):
        """ Determines filename for volcat file based on vdir, date_time and vid"""
        vfiles = '*.nc'
        volclist = glob(self.vdir+vfiles)
        match = self.find_match()
        fname = [f for f in volclist if match in f]
        self.fname = fname[0]
        return self.fname

    def get_area(self, write=False, correct_parallax=True):
        """Calculates the area (km^2) of each volcat grid cell
        Converts degress to meters using a radius of 6378.137km.
        Input:
        write: boolean (default: False) Write area to file
        correct_parallax: boolean (default: True) Use parallax correct lat/lon values
        output:
        area: xarray containing gridded area values
        """
        d2r = pi / 180.0  # convert degress to radians
        d2km = 6378.137 * d2r  # convert degree latitude to kilometers

        if self.fname:
            dset = volcat.open_dataset(self.fname, correct_parallax=correct_parallax)
        else:
            print('ERROR: Need volcat filename!')

        # Extracts ash mass array (two methods - one is smaller domain around feature)
        # Removes time dimension
        mass = dset.ash_mass_loading[0, :, :]
        #mass = volcat.get_mass(dset)[0, :, :]
        if correct_parallax == True:
            lat = mass.latitude.transpose('x', 'y')
            lon = mass.longitude.transpose('x', 'y')
        else:
            lat = mass.latitude
            lon = mass.longitude
        latrad = lat * d2r  # Creating latitude array in radians
        coslat = np.cos(latrad) * d2km * d2km
        # Creates an array copy of mass filled with 0.
        area = xr.full_like(mass, 0.)
        shape = np.shape(area)

        # Begins looping through each element of array
        i = 0
        while i < (shape[0] - 1):
            j = 0
            while j < (shape[1] - 1):
                area[i, j] = abs(lat[i, j] - lat[i+1, j]) * \
                    abs(abs(lon[i, j]) - abs(lon[i, j+1])) * coslat[i, j]
                j += 1
            i += 1
        area.name = 'area'
        area.attrs['long_name'] = 'area of each lat/lon grid box'
        area.attrs['units'] = 'km^2'
        # Reformatting array attributes
        if write == True:
            directory = self.vdir+'Area/'
            match = self.find_match()
            if correct_parallax == True:
                areafname = 'area_'+match+'_pc.nc'
            else:
                areafname = 'area_'+match+'.nc'
            print(directory+areafname)
            area.to_netcdf(directory+areafname)
        return area
