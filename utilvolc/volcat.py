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

# from pyresample.bucket import BucketResampler


"""
This script contains routines that open/read VOLCAT data in xarray format,
manipulate arrays as necessary, and plots desirable variables.
-------------
Functions:
-------------
open_dataset: opens single VOLCAT file
open_hdf: opens single NRT HDF VOLCAT file
create_netcdf: creates netcdf of import variables from NRT HDF
open_mfdataset: opens multiple VOLCAT files
regrid_volcat: regrids with monet.remap_nearest
regrid_volcat2: regrids with monet.remap_xesmf
average_volcat_new: regrids using regrid_volcat() and adds avg mass load, max ash height variables to dataset
average_volcat: averages VOLCAT files over designated time period
get_volcat_name_df: puts parts of volcat file name in pandas dataframe
get_volcat_list: returns list of data-arrays with volcat data
write_regridded_files: writes regridded files
write_parallax_corrected_files: writes new files with parallax corrected lat/lon
find_volcat: finds volcat files in designated directory
test_volcat: tests to see if parallax corrected lat/lon values exist
open_dataset2: opens single volcat file in Reventador format
bbox: finds bounding box around data - used for trimming arrays
_get_time: set time dimension for VOLCAT data
_get_latlon: rename lat/lon, set coordinates of VOLCAT data
get_data: extracts desired data from large volcat file
check_names:
create_pc_plot: plots parallax corrected vs uncorrected values
compare_pc: compares corrected vs uncorrected values
get_pc_latitude: uses parallax corrected latitude
get_pc_longitude: uses parallax corrected longitude
get_height: returns array of ash top height from VOLCAT
get_radius: returns array of ash effective radius from VOLCAT
get_total_mass: returns total mass in volcat file
get_mass:  returns array of ash mass loading from VOLCAT
get_ashdet:
mass_sum:
get_time:
get_atherr: returns array of ash top height error from VOLCAT
plot_height: plots ash top height from VOLCAT
plot_radius: plots ash effective radius from VOLCAT
plot_mass: plots ash mass loading from VOLCAT
plot_gen: generates quick plot, not saved
matchvals:
matchvals2:
find_iii:
correct_pc: corrects parallax
Class: VolcatName
     compare: compares volcat file names - shows difference
     parse: parses name into components
     create_name: in progress
------------
"""


def open_dataset(
    fname,
    gridspace=None,
    correct_parallax=False,
    mask_and_scale=True,
    decode_times=False,
):
    """
    Opens single VOLCAT file
    gridspace: only necessary if doing parallax correction
    mask_and_scale : needs to be set to True for Bezymianny data.
    decode_times   : needs to be True for some of the hdf data.

    """
    # 03/07/2021 The Bezy height data has a fill value of -1,
    # scale_factor of 0.01 and offset of 0.
    # The scale factor needs to be applied to get output in km.

    # ash_mass_loading has no scale_factor of offset and fill value is -999.
    dset = xr.open_dataset(
        fname, mask_and_scale=mask_and_scale, decode_times=decode_times
    )
    # not needed for new Bezy data.
    try:
        dset = dset.rename({"Dim1": "y", "Dim0": "x"})
    except:
        pass
    if "some_vars.nc" in fname:
        pass
    elif "pc.nc" in fname or "rg.nc" in fname:
        return dset
    else:
        # use parallax corrected if available and flag is set.
        dset = _get_latlon(dset, "latitude", "longitude")
        dset = _get_time(dset)
    if "pc_latitude" in dset.data_vars and correct_parallax:
        print("correcting pc")
        if not gridspace:
            dset = correct_pc(dset)
        else:
            dset = correct_pc(dset, gridspace=gridspace)
        dset.attrs.update({"parallax corrected coordinates": "True"})
    elif "pc_latitude" not in dset.data_vars and correct_parallax:
        print("WARNING: cannot correct parallax. Data not found in file")
        dset.attrs.update({"parallax corrected coordinates": "False"})
    else:
        dset.attrs.update({"parallax corrected coordinates": "False"})
    return dset


def open_hdf(fname):
    """Opens single HDF NRT VOLCAT file"""
    # 4/12/2021 Creating reader for HDF files - NRT, not processed
    dset = xr.open_dataset(fname, mask_and_scale=False, decode_times=False)
    dset = dset.rename({"lines": "y", "elements": "x"})
    # create 2d lat lon arrays
    lon2d, lat2d = _make2d_latlon(dset)
    # Rename, add variable and assing to coordinates
    lon = xr.DataArray(lon2d, name="longitude", dims=["y", "x"])
    lat = xr.DataArray(lat2d, name="latitude", dims=["y", "x"])
    attrs = dset.attrs
    dset = xr.merge([dset, lat, lon])
    dset = dset.set_coords(["latitude", "longitude"])
    dset.attrs = attrs
    return dset


def create_netcdf(fname1, fname2):
    """Creates netcdf of important variables from L1 and L2 VOLCAT hdf files
    Writes to same directory as fname2 files"""
    dset1 = xr.open_dataset(fname1, mask_and_scale=False, decode_times=False)
    lat = dset1.pixel_latitude.rename({"lines": "y", "elements": "x"}).rename(
        "latitude"
    )
    lon = dset1.pixel_longitude.rename({"lines": "y", "elements": "x"}).rename(
        "longitude"
    )
    # Ash Top Height, Ash Mass, Ash Effective Radius
    dset2 = xr.open_dataset(fname2, mask_and_scale=False, decode_times=False)
    attrs = dset2.attrs
    namestr = dset2.attrs["Default_Name_ash_ret"]
    mass = (
        dset2[namestr + "_ash_mass_loading"]
        .rename({"lines": "y", "elements": "x"})
        .rename("ash_mass_loading")
    )
    height = (
        dset2[namestr + "_ash_top_height"]
        .rename({"lines": "y", "elements": "x"})
        .rename("ash_cloud_height")
    )
    radius = (
        dset2[namestr + "_ash_effective_radius"]
        .rename({"lines": "y", "elements": "x"})
        .rename("ash_effective_radius")
    )

    # Creating netcdf of important variables
    dset = xr.merge([mass, height, radius, lat, lon])
    dset = dset.set_coords(["latitude", "longitude"])
    dset.attrs = attrs
    dset = _get_time2(dset)
    dset.to_netcdf(fname2[:-11] + "_" + fname2[-10:-3] + "some_vars.nc")
    return print(fname2[:-11] + "_" + fname2[-10:-3] + "some_vars.nc created!")


def open_mfdataset(fname):
    # 12/1/2d020 Not modified for new files (Bezy)
    # TO DO - modify for new files.
    """Opens multiple VOLCAT files"""
    print(fname)
    # dset = xr.open_mfdataset(fname, concat_dim='time', decode_times=False, mask_and_scale=False)
    from glob import glob
    from numpy import sort

    files = sort(glob(fname))
    das = []
    for i in files:
        das.append(open_dataset(i))
    dset = xr.concat(das, dim="time")
    dset = _get_latlon(dset)
    dset = dset.rename({"lines": "y", "elements": "x"})
    return dset


def regrid_volcat(das, cdump):
    """
    Regridding with monet.remap_nearest
    das : list of volcat xarrays
    cdump : xarray DataArray with latitude/longitude to regrid to.
    returns xarray with volcat data with dimension of time and regridded to match cdump.
    """
    # In progress.
    # das is list of volcat datasets.
    # cdump is dataset with appropriate grid.
    # This function maps to new grid.

    # remap_nearest may not be what we want to use. Seems that some small areas with low
    # mass 'disappear' using this regridding scheme. May want to look into using pyresample.bucket or other.
    rai = 1e5
    mlist = []
    hlist = []
    total_mass = []
    feature_area = []
    feature_id = []

    for iii, dset in enumerate(das):
        print("time in loop", dset.time.values)
        near_mass = cdump.monet.remap_nearest(
            dset.ash_mass_loading.isel(time=0), radius_of_influence=rai
        )
        near_height = cdump.monet.remap_nearest(
            dset.ash_cloud_height.isel(time=0), radius_of_influence=rai
        )
        near_mass = near_mass.compute()
        near_height = near_height.compute()
        mlist.append(near_mass)
        hlist.append(near_height)
        total_mass.append(dset.ash_mass_loading_total_mass)
        feature_area.append(dset.feature_area)
    newmass = xr.concat(mlist, dim="time")
    newhgt = xr.concat(hlist, dim="time")
    totmass = xr.concat(total_mass, dim="time")
    farea = xr.concat(feature_area, dim="time")
    dnew = xr.Dataset(
        {
            "ash_mass_loading": newmass,
            "ash_cloud_height": newhgt,
            # 'effective_radius_of_ash': newrad,
            "ash_mass_loading_total_mass": totmass,
            "feature_area": farea,
        }
    )
    # 'feature_area': dset.feature_area,
    # 'feature_age': dset.feature_age,
    # 'feature_id': dset.feature_id})
    # add global attributes.
    dnew = dnew.assign_attrs(dset.attrs)
    dnew.ash_mass_loading.attrs.update(dset.ash_mass_loading.attrs)
    dnew.ash_cloud_height.attrs.update(dset.ash_cloud_height.attrs)
    dnew.time.attrs.update({"standard_name": "time"})
    # propogate attributes on latitude and longitude
    dnew.latitude.attrs.update(dset.latitude.attrs)
    dnew.longitude.attrs.update(dset.longitude.attrs)
    dnew.attrs.update({"Regrid Method": "remap_nearest"})
    return dnew


def regrid_volcat2(das, cdump):
    """
    Regridding with monet.remap_xesmf
    das : list of volcat xarrays
    returns xarray with volcat data with dimension of time and regridded to match cdump.
    """
    # In progress.
    # das is list of volcat datasets.
    # cdump is dataset with appropriate grid.
    # This function maps to new grid.
    # mass 'disappear' using this regridding scheme. May want to look into using pyresample.bucket or other
    mlist = []
    hlist = []
    total_mass = []
    for iii, dset in enumerate(das):
        near_mass = cdump.p006.squeeze().monet.remap_xesmf(
            dset.ash_mass_loading.squeeze(), method="bilinear"
        )
        near_height = cdump.p006.monet.squeeze().remap_xesmf(
            dset.ash_cloud_height.squeeze(), method="bilinear"
        )
        near_mass = near_mass.compute()
        near_height = near_height.compute()
        mlist.append(near_mass)
        hlist.append(near_height)
        total_mass.append(dset.ash_mass_loading_total_mass)
    newmass = xr.concat(mlist, dim="time")
    newhgt = xr.concat(hlist, dim="time")
    totmass = xr.concat(total_mass, dim="time")
    dnew = xr.Dataset(
        {
            "ash_mass_loading": newmass,
            "ash_cloud_height": newhgt,
            # 'effective_radius_of_ash': newrad,
            "ash_mass_loading_total_mass": totmass,
        }
    )
    # 'feature_area': dset.feature_area,
    # 'feature_age': dset.feature_age,
    # 'feature_id': dset.feature_id})
    dnew.time.attrs.update({"standard_name": "time"})
    dnew.latitude.attrs.update({"standard_name": "latitude"})
    dnew.longitude.attrs.update({"standard_name": "longitude"})
    dnew.attrs.update({"Regrid Method": "remap_xesmf bilinear"})
    return dnew


def regrid_volcat_xesmf(das, cdump, method):
    """
    Regridding with xesmf - first gridding irregular volcat grid to
    regular grid (same resolution) using nearest neighbor
    Then gridding to regular cdump grid using conservative method
    das: list of volcat xarrays
    cdump: hysplit cdump xarray
    method: string of possible xesmf regridding techniques
    returns xarray with volcat data with dimension of time and regridded to match cdump.
    """
    import xesmf as xe
    import numpy as np

    # In progress.

    def regrid(ds_source, ds_target, da_source, method):
        # ds_source: dataset of data to be regridded
        # ds_target: dataset of target array
        # da_source: dataarray of data to be regridded
        # NOTE: latitude and longitude must be named lat lon for this to work - use rename
        regridder = xe.Regridder(ds_source, ds_target, method, periodic=True)
        da_target = regridder(da_source)
        regridder.clean_weight_file()
        return da_target

    def rename(xra):
        # Xarray to rename latitude and longitude
        newx = xra.rename({"latitude": "lat", "longitude": "lon"})
        return newx

    def make_grid(xra, d_lon, d_lat):
        # xra: xarray
        # d_lon: delta lon
        # d_lat: delta lat
        xra = rename(xra)
        grid = xe.util.grid_2d(
            np.min(xra.lon) - 1.0,
            np.max(xra.lon) + 1.0,
            d_lon,
            np.min(xra.lat) - 1.0,
            np.max(xra.lat) + 1.0,
            d_lat,
        )
        return grid

    mlist = []
    hlist = []
    total_mass = []
    vgrid = make_grid(das[-1], 0.05, 0.05)
    # vgrid = make_grid(cdump, 0.1, 0.1)
    for iii, dset in enumerate(das):
        dset2 = rename(dset)
        ashmass = regrid(dset2, vgrid, dset.ash_mass_loading, "nearest_s2d")
        height = regrid(dset2, vgrid, dset.ash_cloud_height, "nearest_s2d")
        mlist.append(ashmass)
        hlist.append(height)
        total_mass.append(dset.ash_mass_loading_total_mass)

    mass = xr.concat(mlist, dim="time")
    hgt = xr.concat(hlist, dim="time")
    totmass = xr.concat(total_mass, dim="time")

    # Regridding to cdump array - conservative method needs box bounds
    cgrid = make_grid(cdump, 0.1, 0.1)
    newmass = regrid(vgrid, cgrid, mass, method)
    newhgt = regrid(vgrid, cgrid, hgt, method)
    newmass = mass
    newhgt = hgt

    dnew = xr.Dataset(
        {
            "ash_mass_loading": newmass,
            "ash_cloud_height": newhgt,
            # 'effective_radius_of_ash': newrad,
            "ash_mass_loading_total_mass": totmass,
        }
    )
    # 'feature_area': dset.feature_area,
    # 'feature_age': dset.feature_age,
    # 'feature_id': dset.feature_id})
    dnew.time.attrs.update({"standard_name": "time"})
    return dnew


def average_volcat_new(das, cdump, skipna=False, convert_nans=False):
    # STILL IN PROGRESS
    """
    Very similar to average_volcat() except it regrids using regrid_volcat()
    Output contains full array from regrid_volcat() with average/maximum
    arrays added

    Inputs:
    das: list of volcat datasets
    cdump: xarray of target grid (hysplit cdump usually)
    skipna: boolean - skip nans when taking mean/max
    convert_nans: boolean - convert nans to 0. in all xarrays BEFORE regridding
    outputs:
    dsetnew: dataset with added ash mass mean/ash height max
    """
    fill = "nan"
    if convert_nans:
        fill = "No fill value"
        dset = []
        i = 0
        while i < len(das):
            dset.append(das[i].fillna(0.0))
            i += 1
        hxr = cdump.fillna(0)
    else:
        dset = das
        hxr = cdump
    dnew = regrid_volcat(dset, hxr)

    avgmass = dnew.ash_mass_loading.mean(dim="time", skipna=skipna, keep_attrs=True)
    maxhgt = dnew.ash_cloud_height.max(dim="time", skipna=skipna, keep_attrs=True)
    # renaming variable
    avgmass = avgmass.load().rename("ash_mass_avg")
    maxhgt = maxhgt.load().rename("ash_height_max")
    # Adding time dimension, changing long_name
    avgmass = avgmass.assign_coords(time=dnew.time[-1]).expand_dims("time")
    maxhgt = maxhgt.assign_coords(time=dnew.time[-1]).expand_dims("time")
    avgmass.attrs[
        "long_name"
    ] = "Average total column loading of ash in the highest continuous ash layer for the previous hour"
    avgmass.attrs["fill_value"] = fill
    maxhgt.attrs[
        "long_name"
    ] = "Maximum cloud top height of the highest continuous ash layer for the previous hour"
    maxhgt.attrs["fill_value"] = fill

    # Merging datasets
    dsetnew = xr.merge([dnew, avgmass, maxhgt], combine_attrs="drop_conflicts")

    return dsetnew


def average_volcat(das, cdump, skipna=False, convert_nans=False):
    # In progress.
    """
    Function first regrids to new grid, then calculates then average
    mass loading and maximum ash height.

    Inputs:
    das: list of volcat datasets
    cdump: is dataset with appropriate grid
    skipna: boolean - flag to skip nan values in grid when calculating mean or max
    convert_nans: noolean - flag to convert nans in datasets to 0
    Output:
    avgmass: mean of volcat ash mass loading, from das
    maxhgt: maximum of volcat ash cloud height

    Notes:
    remap_nearest may not be what we want to use.
    Seems that some small areas with low
    mass 'disappear' using this regridding scheme.
    May want to look into using pyresample.bucket or other.
    """
    rai = 1e5
    mlist = []
    hlist = []
    for iii, dset in enumerate(das):
        near_mass = cdump.monet.remap_nearest(
            dset.ash_mass_loading.isel(time=0), radius_of_influence=rai
        ).load()
        near_height = cdump.monet.remap_nearest(
            dset.ash_cloud_height.isel(time=0), radius_of_influence=rai
        ).load()
        mlist.append(near_mass)
        hlist.append(near_height)
    newmass = xr.concat(mlist, dim="time")
    newhgt = xr.concat(hlist, dim="time")
    # when averaging the mass need to convert nan's to zero?
    if convert_nan:
        newmass = newmass.fillna(0.0)
        newhgt = newhgt.fillna(0.0)
    # option to skip nans
    avgmass = newmass.mean(dim="time", skipna=skipna)
    # note that averaging the height is not correct, better to take maximum along time
    maxhgt = newhgt.max(dim="time", skipna=skipna)
    return avgmass, maxhgt


def get_volcat_name_df(tdir, daterange=None, vid=None, fid=None, include_last=False):
    """
    Returns dataframe with columns being the information in the vhash
    dictionary of the VolcatName class. This is all the information collected from the filename.
    """
    tlist = find_volcat(tdir, vid=None, daterange=None, return_val=2)
    vlist = [x.vhash for x in tlist]
    temp = pd.DataFrame(vlist)
    if isinstance(daterange, (list, np.ndarray)):
        temp = temp[temp["edate"] >= daterange[0]]
        if include_last:
            temp = temp[temp["edate"] <= daterange[1]]
        else:
            temp = temp[temp["edate"] < daterange[1]]
    if vid:
        temp = temp[temp["volcano id"] == vid]
    if fid:
        temp = temp[temp["fid"] == fid]

    if "fid" in temp.columns:
        temp = temp.sort_values(["volcano id", "fid", "edate"], axis=0)
    else:
        temp = temp.sort_values(["volcano id", "edate"], axis=0)
    return temp


# two json files.
# the first one


def read_event_summary():
    # list of all active eruptions in the satellite image.
    # Return dataframe made from dictionary in the event summary.
    return df


def summarize_files(volcat_event_df):
    """
    what volcano id's are available.
    what time periods.
    """
    # return volcano id's.
    return -1


def choose_files(volcat_event_df, vid, frequency=10):
    """
    volcat_event_df with columns specified by VolcatName dictionary.
    frequency: how far apart should files to be spaced (minutes)
    """
    return -1


def get_volcat_list(
    tdir,
    daterange=None,
    vid=None,
    fid=None,
    flist=None,
    return_val=2,
    correct_parallax=True,
    mask_and_scale=True,
    decode_times=True,
    verbose=False,
    include_last=True,
):
    """
    returns list of data-arrays with volcat data.
    Inputs:
    tdir: string - directory of volcat files
    daterange: datetime object -  [datetime0, datetime1] or none
    vid: string - volcano ID
    return_val: integer (1,2,3) - see find_volcat() for explanation
    correct_parallax: boolean
    mask_and_scale: boolean
    decode_times: boolean
    verbose: boolean
    include_last: boolean
    Outputs:
    das: list of datasets
    """
    if flist:
        filnames = flist
    else:
        tframe = get_volcat_name_df(
            tdir, vid=vid, fid=fid, daterange=daterange, include_last=include_last
        )
        filenames = tframe.filename.values
    das = []
    for iii in filenames:
        # opens volcat files using volcat.open_dataset
        if not "_pc" in iii:
            das.append(
                open_dataset(
                    os.path.join(tdir, iii),
                    correct_parallax=correct_parallax,
                    mask_and_scale=mask_and_scale,
                    decode_times=decode_times,
                )
            )
        else:
            das.append(xr.open_dataset(os.path.join(tdir, iii)))
    return das


def write_regridded_files(
    cdump, tdir, wdir, tag="rg", vid=None, daterange=None, verbose=False
):
    """
    cdump : xarray DataArray with latitude and longitude grid for regridding.
    tdir : str : location of volcat files.
    wdir : str : location to write new files
    vid : volcano id : if None will find all
    daterange : [datetime, datetime] : if None will find all.
    verbose: boolean
    tag: used to create filename of new file.

    creates netcdf files regridded values.
    files have same name with _{tag}.nc added to the end.
    Current convention is to use tag=pc.
    These will be needed for input into MET.

    Currently no overwrite option exists in this function. If the file
    already exists, then this function returns a message to that effect and
    does not overwrite the file.
    """
    vlist = find_volcat(tdir, vid, daterange, verbose=verbose, return_val=2)
    for iii, val in enumerate(vlist):
        fname = val.fname
        new_fname = fname.replace(".nc", "_{}.nc".format(tag))
        if os.path.isfile(os.path.join(wdir, new_fname)):
            print(
                "Netcdf file exists {} in directory {} cannot write ".format(
                    new_fname, wdir
                )
            )
        else:
            if verbose:
                print("writing {} to {}".format(new_fname, wdir))
            dset = open_dataset(
                os.path.join(tdir, fname), correct_parallax=False, decode_times=True
            )
            dnew = regrid_volcat([dset], cdump)
            dnew.to_netcdf(os.path.join(wdir, new_fname))


def write_parallax_corrected_files(
    tdir,
    wdir,
    vid=None,
    daterange=None,
    verbose=False,
    flist=None,
    gridspace=None,
    tag="pc",
):
    """
    ***If flist is not specified, this does not work. There are folders in the tdir and they
    cause a problem with the function. Flist must not include directories, just file names***
    tdir : str : location of volcat files.
    wdir : str : location to write new files
    vid : volcano id : if None will find all
    daterange : [datetime, datetime] : if None will find all.
    verbose: boolean
    flist: list of files? ***NEED TO SPECIFY FILE LIST***
    gridspace: float : grid size of pc array
    tag: used to create filename of new file.

    creates netcdf files with parallax corrected values.
    files have same name with _{tag}.nc added to the end.
    Current convention is to use tag=pc.
    These will be needed for input into MET.

    Currently no overwrite option exists in this function. If the file
    already exists, then this function returns a message to that effect and
    does not overwrite the file.
    """
    if not flist:
        vlist = find_volcat(tdir, vid, daterange, verbose=verbose, return_val=2)
    else:
        vlist = flist
    for iii, val in enumerate(vlist):
        if isinstance(val, str):
            fname = val
        else:
            fname = val.fname
        new_fname = fname.replace(".nc", "_{}.nc".format(tag))
        if os.path.isfile(os.path.join(wdir, new_fname)):
            print(
                "Netcdf file exists {} in directory {} cannot write ".format(
                    new_fname, wdir
                )
            )
        else:
            if verbose:
                print("writing {} to {}".format(new_fname, wdir))
            dset = open_dataset(
                os.path.join(tdir, fname), gridspace=gridspace, correct_parallax=True
            )
            dset.to_netcdf(os.path.join(wdir, new_fname))


def find_volcat(
    tdir, vid=None, daterange=None, return_val=2, verbose=False, include_last=False
):
    ##NOT WORKING FOR NISHINOSHIMA DATA##
    """
    Locates files in tdir which follow the volcat naming
    convention as defined in VolcatName class.
    If a daterange is defined will return only files

    Inputs:
    tdir : string - volcat files directory
    vid: string - volcano id
    daterange : [datetime, datetime] or None
    include_last : boolean
               True - includes volcat data with date = daterange[1]
               False - only include data with date < daterange[1]
    return_val : integer
               1 - returns dictionary
               2-  returns list of VolcatName objects.
               3 - returns list of filenames
    Returns:
               1 - returns dictionary. key is date. values is VolcatName object.
               2 - returns list of VolcatName objects.
               3 - returns list of filenames
    """
    import sys

    vhash = {}  # dictionary
    nflist = []  # list of filenames
    vnlist = []  # list of filenames
    if not os.path.isdir(tdir):
        print("directory not valid {}".format(tdir))
    for fln in os.listdir(tdir):
        try:
            vn = VolcatName(fln)
        except:
            if verbose:
                print("Not VOLCAT filename {}".format(fln))
            continue
        if daterange and include_last:
            if vn.date < daterange[0] or vn.date > daterange[1]:
                if verbose:
                    print("date not in range", vn.date, daterange[0], daterange[1])
                continue
        elif daterange and not include_last:
            if vn.date < daterange[0] or vn.date >= daterange[1]:
                if verbose:
                    print("date not in range", vn.date, daterange[0], daterange[1])
                continue
        if vid and vn.vhash["volcano id"] != vid:
            continue
        if return_val == 1:
            if vn.date not in vhash.keys():
                vhash[vn.date] = vn
            else:
                print("two files with same date")
                print(vhash[vn.date].compare(vn))
        elif return_val == 2:
            vnlist.append(vn)
        elif return_val == 3:
            nflist.append(fln)
    if return_val == 1:
        return vhash
    elif return_val == 2:
        return vnlist
    elif return_val == 3:
        return nflist


def test_volcat(tdir, daterange=None, verbose=True):
    """
    checks the pc_latitude field for values greater than 0.
    """
    vnlist = find_volcat(tdir, daterange, verbose)
    for key in vnlist.keys():
        vname = vnlist[key].fname
        dset = open_dataset(os.path.join(tdir, vname), pc_correct=False)
        if np.max(dset.pc_latitude) > 0:
            print("passed")
        else:
            print("failed")


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

    def __init__(self, fname):
        # if full directory path is input then just get the filename
        if "/" in fname:
            temp = fname.split("/")
            self.fname = temp[-1]
        else:
            self.fname = fname
        self.vhash = {}
        self.date = None
        self.dtfmt = "s%Y%j_%H%M%S"
        self.image_dtfmt = "b%Y%j_%H%M%S"

        self.keylist = ["algorithm name"]
        self.keylist.append("satellite platform")
        self.keylist.append("event scanning strategy")
        self.keylist.append("event date")
        self.keylist.append("event time")
        self.keylist.append("fid")
        self.keylist.append("volcano id")
        self.keylist.append("description")
        self.keylist.append("WMO satellite id")
        self.keylist.append("image scanning strategy")
        self.keylist.append("image date")
        self.keylist.append("image time")
        self.keylist.append("feature id")

        self.pc_corrected = False
        self.parse(self.fname)
        self.vhash["filename"] = fname

    def __lt__(self, other):
        """
        sort by
        volcano id first.
        date
        feature id if it exists.
        """
        if self.vhash["volcano id"] < other.vhash["volcano id"]:
            return True
        if "fid" in self.vhash.keys() and "fid" in other.vhash.keys():
            if self.vhash["fid"] < other.vhash["fid"]:
                return True
        if self.date < other.date:
            return True
        if self.image_date < other.image_date:
            return True
        sortlist = [
            "feature id",
            "image scanning strategy",
            "WMO satellite id",
            "description",
            "event scanning strategy",
            "satellite platform",
            "algorithm name",
        ]
        for key in sortlist:
            if key in other.vhash.keys() and key in self.vhash.keys():
                if self.vhash[key] < other.vhash[key]:
                    return True

    def compare(self, other):
        """
        other is another VolcatName object.
        Returns
        dictionary of information which is different.
        values is a  tuple of (other value, self value).
        """
        diffhash = {}
        for key in self.keylist:
            if key in other.vhash.keys() and key in self.vhash.keys():
                if other.vhash[key] != self.vhash[key]:
                    diffhash[key] = (other.vhash[key], self.vhash[key])
        return diffhash

    def __str__(self):
        val = [self.vhash[x] for x in self.keylist]
        return str.join("_", val)

    def parse(self, fname):
        temp = fname.split("_")
        if "pc" in temp[-1]:
            self.pc_corrected = True
        jjj = 0
        for iii, key in enumerate(self.keylist):
            val = temp[jjj]
            # nishinoshima files have a g00? code before the volcano id.
            if key == "fid":
                if val[0] == "g":
                    self.vhash[key] = val
                else:
                    continue
            self.vhash[key] = val
            jjj += 1
        # Event date marks date of the data collection
        dstr = "{}_{}".format(self.vhash[self.keylist[3]], self.vhash[self.keylist[4]])
        self.date = datetime.datetime.strptime(dstr, self.dtfmt)
        # Image date is ?
        dstr = "{}_{}".format(
            self.vhash[self.keylist[10]], self.vhash[self.keylist[11]]
        )
        self.image_date = datetime.datetime.strptime(dstr, self.image_dtfmt)
        self.vhash[self.keylist[11]] = self.vhash[self.keylist[11]].replace(".nc", "")

        self.vhash["idate"] = self.image_date
        self.vhash["edate"] = self.date

        return self.vhash

    def create_name(self):
        """
        To do: returns filename given some inputs.
        """
        return -1


def open_dataset2(fname):
    """Opens single VOLCAT file in reventador format"""
    print(fname)
    dset = xr.open_dataset(fname, mask_and_scale=False, decode_times=False)
    # dset = dset.rename({"Dim1":'y',"Dim0":'x'})
    # dset = _get_latlon(dset)
    # dset = _get_time(dset)
    return dset


def bbox(darray, fillvalue):
    """Returns bounding box around data
    Input: Must be dataarray
    Outupt: Lower left corner, upper right corner of bounding box
    around data.
    if fillvalue is None then assume Nan's.
    """
    arr = darray[0, :, :].values
    if fillvalue:
        a = np.where(arr != fillvalue)
    else:
        a = np.where(~np.isnan(arr))
    if np.min(a[0]) != 0.0 and np.min(a[1]) != 0.0:
        bbox = (
            [np.min(a[0] - 3), np.min(a[1]) - 3],
            [np.max(a[0] + 3), np.max(a[1]) + 3],
        )
    else:
        bbox = ([np.min(a[0]), np.min(a[1])], [np.max(a[0]), np.max(a[1])])
    return bbox


def _get_latlon(dset, name1="latitude", name2="longitude"):
    dset = dset.set_coords([name1, name2])
    return dset


def _make2d_latlon(dset):
    lon = np.linspace(
        dset.attrs["Longitude_Range"][0],
        dset.attrs["Longitude_Range"][1],
        dset.attrs["Last_Element_Processed"],
    )
    lat = np.linspace(
        dset.attrs["Latitude_Range"][1],
        dset.attrs["Latitude_Range"][0],
        dset.attrs["Line_Segment_Size"],
    )
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d, lat2d


def _get_time(dset):
    import pandas as pd

    temp = dset.attrs["time_coverage_start"]
    time = pd.to_datetime(temp)
    dset["time"] = time
    dset = dset.expand_dims(dim="time")
    dset = dset.set_coords(["time"])
    return dset


def _get_time2(dset):
    import pandas as pd

    date = "20" + str(dset.attrs["Image_Date"])[1:]
    time1 = str(dset.attrs["Image_Time"])
    if len(time1) == 5:
        time1 = "0" + str(dset.attrs["Image_Time"])
    time = pd.to_datetime(date + time1, format="%Y%j%H%M%S", errors="ignore")
    dset["time"] = time
    dset = dset.expand_dims(dim="time")
    dset = dset.set_coords(["time"])
    return dset


# Extracting variables


def get_data(dset, vname, clip=True):
    gen = dset.data_vars[vname]
    atvals = gen.attrs
    fillvalue = None
    if "_FillValue" in gen.attrs:
        fillvalue = gen._FillValue
        gen = gen.where(gen != fillvalue)
        fillvalue = None
    if clip:
        box = bbox(gen, fillvalue)
        gen = gen[:, box[0][0] : box[1][0], box[0][1] : box[1][1]]
        if "_FillValue" in gen.attrs:
            gen = gen.where(gen != fillvalue)
        else:
            gen = gen.where(gen)
    # applies scale_factor and offset if they are in the attributes.
    if "scale_factor" in gen.attrs:
        gen = gen * gen.attrs["scale_factor"]
    if "offset" in gen.attrs:
        gen = gen + gen.attrs["offset"]
    if "add_offset" in gen.attrs:
        gen = gen + gen.attrs["add_offset"]
    # keep relevant attributes.
    new_attr = {}
    for key in atvals.keys():
        if key not in ["_FillValue", "add_offset", "offset", "scale_factor"]:
            new_attr[key] = atvals[key]
    gen.attrs = new_attr
    return gen


def check_names(dset, vname, checklist, clip=True):
    if vname:
        return get_data(dset, vname, clip=clip)
    for val in checklist:
        if val in dset.data_vars:
            return get_data(dset, val, clip=clip)
    return xr.DataArray()


def create_pc_plot(dset):
    """
    creates plots of parallax corrected vs. uncorrected values.
    """

    def subfunc(ax, vals):
        ax.plot(vals[0], vals[1], "k.", MarkerSize=1)
        # plot 1:1 line
        minval = np.min(vals[0])
        maxval = np.max(vals[0])
        ax.plot([minval, maxval], [minval, maxval], "--r.", MarkerSize=1)

    latitude, longitude = compare_pc(dset)
    fig = plt.figure(1)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.set_ylabel("uncorrected")
    ax2.set_ylabel("uncorrected")
    ax2.set_xlabel("corrected")

    subfunc(ax1, latitude)
    subfunc(ax2, longitude)
    return fig, ax1, ax2


def compare_pc(dset):
    """
    Returns:
    latitude : [list of parrallax corrected values, list of uncorrected values]
    longitude : [list of parrallax corrected values, list of uncorrected values]
    """

    def process(pc, val):
        # pair corrected and uncorrected values.
        pzip = list(zip(pc, val))
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
    longitude = process(pclon, lonvals)
    return latitude, longitude


def get_pc_latitude(dset, vname=None, clip=True):
    """Returns array with retrieved height of the highest layer of ash."""
    """Default units are km above sea-level"""
    checklist = ["pc_latitude"]
    return check_names(dset, vname, checklist, clip=clip)


def get_pc_longitude(dset, vname=None, clip=True):
    """Returns array with retrieved height of the highest layer of ash."""
    """Default units are km above sea-level"""
    checklist = ["pc_longitude"]
    return check_names(dset, vname, checklist, clip=clip)


def get_height(dset, vname=None, clip=True):
    """Returns array with retrieved height of the highest layer of ash.
    Default units are km above sea-level"""
    checklist = ["ash_cth", "ash_cloud_height"]
    return check_names(dset, vname, checklist, clip=clip)


def get_radius(dset, vname=None, clip=True):
    """Returns 2d array of ash effective radius
    Default units are micrometer"""
    checklist = ["ash_r_eff", "effective_radius_of_ash"]
    return check_names(dset, vname, checklist, clip=clip)


def get_total_mass(dset):
    # unit is in Tg.
    """Units are in Tg"""
    return dset.ash_mass_loading_total_mass.values[0]


def get_mass(dset, vname=None, clip=True):
    """Returns 2d array of ash mass loading
    Default units are grams / meter^2"""
    checklist = ["ash_mass", "ash_mass_loading"]
    return check_names(dset, vname, checklist, clip=clip)


def get_ashdet(dset, vname=None, clip=True):
    """Returns 2d array of detected ash
    Values > 0.0 = detected ash
    Values < 0.0 = no detected ash
    Can be used to determine if ash was detected, but ash mass or ash height was not"""
    checklist = ["ash_spectral_signature_strength"]
    return check_names(dset, vname, checklist, clip=clip)


def mass_sum(dset):
    mass = get_mass(dset)
    mass2 = mass.where(mass > 0.0, 0.0).values
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


def plot_height(dset):
    """Plots ash top height from VOLCAT
    Does not save figure - quick image creation"""
    fig = plt.figure("Ash_Top_Height")
    title = "Ash Top Height (km)"
    ax = fig.add_subplot(1, 1, 1)
    plot_gen(dset, ax, val="height", time=None, plotmap=True, title=title)


def plot_radius(dset):
    """Plots ash effective radius from VOLCAT
    Does not save figure - quick image creation"""
    fig = plt.figure("Ash_Effective_Radius")
    title = "Ash effective radius ($\mu$m)"
    ax = fig.add_subplot(1, 1, 1)
    plot_gen(dset, ax, val="radius", time=None, plotmap=True, title=title)


def plot_mass(dset):
    fig = plt.figure("Ash_Mass_Loading")
    ax = fig.add_subplot(1, 1, 1)
    plot_gen(dset, ax, val="mass", time=None, plotmap=True, title="Ash_Mass_Loading")


def plot_gen(dset, ax, val="mass", time=None, plotmap=True, title=None):
    """Plot ash mass loading from VOLCAT
    Does not save figure - quick image creation"""
    # lat=dset.latitude
    # lon=dset.longitude
    if val == "mass":
        mass = get_mass(dset)
    elif val == "radius":
        mass = get_radius(dset)
    elif val == "height":
        mass = get_height(dset)
    if time and "time" in mass.coords:
        mass = mass.sel(time=time)
    elif "time" in mass.coords:
        mass = mass.isel(time=0)
    lat = mass.latitude
    lon = mass.longitude
    if plotmap:
        m = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        m.add_feature(cfeat.LAND)
        m.add_feature(cfeat.COASTLINE)
        m.add_feature(cfeat.BORDERS)
        plt.pcolormesh(lon, lat, mass, transform=ccrs.PlateCarree())
    else:
        plt.pcolormesh(lon, lat, mass)
    plt.colorbar()
    plt.title(title)
    plt.show()


def matchvals(pclat, pclon, massra, height):
    # pclat : xarray DataArray
    # pclon : xarray DataArray
    # mass : xarray DataArray
    # height : xarray DataArray
    # used in correct_pc
    # returns 1D list of tuples of values in the 4 DataArrays
    pclon = pclon.values.flatten()
    pclat = pclat.values.flatten()
    mass = massra.values.flatten()
    height = height.values.flatten()
    tlist = list(zip(pclat, pclon, mass, height))
    # only return tuples in which mass has a valid value
    if "_FillValue" in massra.attrs:
        fill = massra.attrs["_FillValue"]
        tlist = [x for x in tlist if x[2] != fill]
    else:
        # get rid of Nans.
        tlist = [x for x in tlist if ~np.isnan(x[2])]
    return tlist


def matchvals2(pclat, pclon, ashdet):
    # pclat : xarray DataArray
    # pclon : xarray DataArray
    # ashdet : xarray DataArray
    # used in correct_pc
    # returns 1D list of tuples of values in the 3 DataArrays
    pclon = pclon.values.flatten()
    pclat = pclat.values.flatten()
    ash = ashdet.values.flatten()
    tlist = list(zip(pclat, pclon, ash))
    # only return tuples in which mass has a valid value
    if "_FillValue" in ashdet.attrs:
        fill = ashdet.attrs["_FillValue"]
        tlist = [x for x in tlist if x[2] != fill]
    else:
        # get rid of Nans.
        tlist = [x for x in tlist if ~np.isnan(x[2])]
    return tlist


def find_iii(tlist, match):
    for iii, val in enumerate(tlist):
        if val == match:
            return iii
    return -1


def determine_pc_grid_space():
    return 1


def correct_pc(dset, gridspace=None):
    """
    moves mass and height values into the coordinate values closest
    to the parallax corrected values. Results in dataset with mass and height shifted
    to parallax corrected positions.
    """
    # 06/02/2021 amc commented out use of the ashdet field.
    # AMR: Added ability to grid parallax corrected data to regular grid

    mass = get_mass(dset, clip=False)
    height = get_height(dset, clip=False)
    effrad = get_radius(dset, clip=False)
    # ashdet = get_ashdet(dset, clip=False)

    if not gridspace:
        newmass = xr.zeros_like(mass.isel(time=0))
        newhgt = xr.zeros_like(height.isel(time=0))
        newrad = xr.zeros_like(effrad.isel(time=0))
    # newashdet = xr.zeros_like(ashdet.isel(time=0))
    else:
        # AMR: Making regular grid for parallax correction
        # Changes pushed 8/26/2021
        # Mass, height, and effrad should all have the same lat/lon
        # if this is not the case, need to adjust this portion of the function
        # maybe loop through the variables to calculate max,min?
        # NEED to make this more general, for various grid spaces
        latmin = round(mass.latitude.values.min())
        latmax = round(mass.latitude.values.max()) + 1.0
        lonmin = round(mass.longitude.values.min())
        lonmax = round(mass.longitude.values.max()) + 1.0
        # lats = np.arange(latmin, latmax, gridspace)
        lats = np.arange(latmax, latmin, gridspace * -1)
        lons = np.arange(lonmin, lonmax, gridspace)
        longitude, latitude = np.meshgrid(lons, lats)
        tmp = np.zeros_like(latitude)
        # Making zeros like arrays
        das = xr.DataArray(
            data=tmp,
            dims=["y", "x"],
            coords=dict(
                latitude=(["y", "x"], latitude), longitude=(["y", "x"], longitude)
            ),
        )
        newmass = das
        newmass.attrs = mass.attrs
        newhgt = das
        newhgt.attrs = height.attrs
        newrad = das
        newrad.attrs = effrad.attrs
        # END of Additional code - AMR

    time = mass.time
    pclat = get_pc_latitude(dset, clip=False)
    pclon = get_pc_longitude(dset, clip=False)
    tlist = np.array(matchvals(pclon, pclat, mass, height))
    # tlist = np.array(matchvals2(pclon, pclat, ashdet))

    indexlist = []
    prev_point = 0
    for point in tlist:
        iii = newmass.monet.nearest_ij(lat=point[1], lon=point[0])
        if iii in indexlist:
            print("WARNING: correct_pc function: some values mapped to same point")
            print(iii, point)
            vpi = find_iii(indexlist, iii)
            print(tlist[vpi])
            # AMR: 9/1/2021
            # Need to add mass from values mapped to same grid point (conserve mass)
            # Take max height from values mappend to same grid point (conserve top height)
            totmass = tlist[vpi][2] + point[2]
            maxhgt = np.max([tlist[vpi][3], point[3]])
            # Reassigning values in point array for mass and height
            point[2] = totmass
            point[3] = maxhgt
            print("Total mass, max height: ", point)
            # End of AMR additions
        newmass = xr.where(
            (newmass.coords["x"] == iii[0]) & (newmass.coords["y"] == iii[1]),
            point[2],
            newmass,
        )
        newhgt = xr.where(
            (newhgt.coords["x"] == iii[0]) & (newhgt.coords["y"] == iii[1]),
            point[3],
            newhgt,
        )
        # AMR: Need to adjust this for effective radius - not currently in tlist, and therefore not in iii
        newrad = xr.where(
            (newrad.coords["x"] == iii[0]) & (newrad.coords["y"] == iii[1]),
            point[3],
            newrad,
        )
        # keeps track of new indices of lat lon points.
        indexlist.append(iii)
        prev_point = point
    # check if any points are mapped to the same point.
    if len(indexlist) != len(list(set(indexlist))):
        print("WARNING: correct_pc function: some values mapped to same point")
        print(len(indexlist), len(list(set(indexlist))))
    # TODO currently the fill value is 0.
    # possibly change to nan or something else?
    newmass = newmass.assign_attrs({"_FillValue": 0})
    newhgt = newhgt.assign_attrs({"_FillValue": 0})
    newrad = newrad.assign_attrs({"_FillValue": 0})

    newmass = newmass.expand_dims("time")
    newhgt = newhgt.expand_dims("time")
    newrad = newrad.expand_dims("time")

    newmass = newmass.transpose("time", "y", "x", transpose_coords=True)
    newhgt = newhgt.transpose("time", "y", "x", transpose_coords=True)
    newrad = newrad.transpose("time", "y", "x", transpose_coords=True)
    # keep original names for mass and height.
    dnew = xr.Dataset(
        {
            "ash_mass_loading": newmass,
            "ash_cloud_height": newhgt,
            "effective_radius_of_ash": newrad,
            "ash_mass_loading_total_mass": dset.ash_mass_loading_total_mass,
            "feature_area": dset.feature_area,
            "feature_age": dset.feature_age,
            "feature_id": dset.feature_id,
        }
    )
    dnew.ash_mass_loading.attrs.update(dset.ash_mass_loading.attrs)
    dnew.ash_cloud_height.attrs.update(dset.ash_cloud_height.attrs)
    dnew.effective_radius_of_ash.attrs.update(dset.effective_radius_of_ash.attrs)
    dnew.time.attrs.update({"standard_name": "time"})
    dnew.latitude.attrs.update({"standard_name": "latitude"})
    dnew.longitude.attrs.update({"standard_name": "longitude"})
    dnew = dnew.assign_attrs(dset.attrs)
    return dnew
