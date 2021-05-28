from matplotlib.colors import BoundaryNorm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import numpy.ma as ma
import monetio.models.hysplit as hysplit
import datetime
import seaborn as sns
import os
import matplotlib.pyplot as plt
from utilhysplit.evaluation.statmain import cdf

"""
FUNCTIONS
The functions in this file are for
manipulating concentration xarray objects which have dimensions of lat,lon,z,time,ens,source
manipulating massloading   xarray objects which have dimensions of lat,lon,time,ens,source


maxconc   : returns maximum concentration along a dimension(x,y, or z)
topheight : returns height of level
ATL       : applied threshold level
ens_mean  : creates array with ensemble mean values.
make_ATL_hysp:

PLOTTING
plotmaxconc : creates plot of maximum concentrations along a dimension (x,y,z)
getclrslevs :
plotATL :
plot_ens_mean :
ashcmap :
draw_map :

For reading writing netcdf file
makenc :
readnc :

Use Examples
maketestra :
exampleATL
"""

def listvals(dra):
    """
    returns 1d list of values in the data-array
    """
    vals = dra.values
    vshape = np.array(vals.shape)
    return vals.reshape(np.prod(vshape))

def ens_cdf(indra, 
            enslist=None, 
            sourcelist=None, 
            timelist=None, 
            threshold=0, 
            plot=True,
            pixel_match=False):
    """
    produces plots of cumulative distribution functions.
    indra : xarray DataArray produced by combine_dataset function or hysp_massload function..
    timelist : list of times in the time coordinate to produce plots for
    threshold : float : produce CDF for values > threshold.
    pixel_match : not working yet.
    Returns:
    cdfhash : dictionary. key is the time. value is a tuple of the CDF (x,y)
    """
    # select sources of interest and stack
    
    if 'source' in indra.dims:
        sourcekey = True
    else:
        sourcekey = False     
    dra = choose_and_stack(indra, enslist, sourcelist)
    cdfhash = {}
    for iii, ens in enumerate(dra.ens.values):
        subdra = dra.sel(ens=ens)
        if not isinstance(timelist, list) and not isinstance(timelist, np.ndarray):
            timelist = subdra.time.values
        for tm in timelist:
            tvals = subdra.sel(time=tm)
            # create cdf from values above threshold
            sdata, y = cdf([x for x in listvals(tvals) if x > threshold])
            if sourcekey:
               key = (tm, ens[0], ens[1])
            else:
               key = (tm, ens)
            cdfhash[key] = (sdata, y)
    if plot:
        plot_cdf(cdfhash)
    return cdfhash

def plot_cdf(cdfhash, fignum=1):
    """
    Plots output from ens_cdf
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    clrs = ['r', 'y', 'g', 'c', 'b', 'k']
    for iii, key in enumerate(cdfhash.keys()):
        if iii > len(clrs)-1:
            clrs.extend(clrs)
        ax.step(cdfhash[key][0], cdfhash[key][1], '-'+clrs[iii])
    ax.set_xscale('log')
    return ax


def draw_map(fignum, fs=20):
    proj = ccrs.PlateCarree()
    fig = plt.figure(fignum, figsize=(16, 9))

    ax = fig.add_subplot(1, 1, 1)
    ax = plt.axes(projection=proj)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray")
    gl.labels_right = False
    gl.labels_top = False
    # ocean = cfeature.NaturalEarthFeature(category='physical',name='ocean',
    #        scale='50m', facecolor = cfeature.COLORE['water'])

    # ax.add_feature(states.edgecolor='gray')
    ax.add_feature(cfeature.LAND)
    return ax


def exampleATL(revash):
    """
    output of maketestra can be used.
    """
    # if not revash:
    #    revash = maketestra()
    enslist = revash.ens.values
    d1 = datetime.datetime(2019, 2, 25, 20)
    level = revash.z.values
    # remove deposition layer if it exists.
    if 0 in level:
        level = np.delete(level, np.where(level=0))
    # mult factor should convert unit mass into mg.
    mult = 1e20
    fignum = 1
    fig = plt.figure(fignum)
    ax = draw_map(fignum)
    plotATL(ax, revash*mult, time=d1, enslist=enslist, level=level)


def maketestra():
    flist = []
    dname = '/pub/ECMWF/JPSS/reventador/ens4/'
    basefname = 'cdump.004a.A'
    elist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    source = 'S1'
    for eee in elist:
        fname = basefname + 'e' + str(eee)
        flist.append((os.path.join(dname, fname), source, '4ERA5e' + str(eee)))
    # revash = monetio.models.hysplit.combine_datatset(flist, drange=None)
    revash = hysplit.combine_dataset(flist)
    return revash


def makenc(dset, ncname):
    """
    time, x, y, z have attributes which cause trouble
    when trying to write a netcdf file.
    """
    attlist = ['area', 'proj4_srs']
    for att in attlist:
        try:
            del dset.time.attrs[att]
        except:
            pass
        try:
            del dset.x.attrs[att]
        except:
            pass
        try:
            del dset.y.attrs[att]
        except:
            pass
        try:
            del dset.z.attrs[att]
        except:
            pass
    dset.to_netcdf(ncname, format='NETCDF4')
    return -1


def readnc(name):
    """
    reads in as dataset.
    convert to dataarray
    remove variable as a dimension
    """
    dset = xr.open_dataset(name)
    dra = dset.to_array()
    dra = dra.isel(variable=0)
    return dra


def ashcmap(t=1):
    """
    possible colormaps for plotting volcanic ash.
    """
    import matplotlib
    # cmap = matplotlib.cm.cubehelix(np.linspace(0,1,20))
    # cmap = matplotlib.cm.RdGy(np.linspace(0,1,20))
    # cmap = matplotlib.colors.ListedColormap(cmap[10:,:-1])
    if t == 1:
        cmap = 'RdGy_r'
    elif t == 2:
        cmap = sns.diverging_palette(200, 1, as_cmap=True, center='light', l=10)
    elif t == 3:
        cmap = sns.diverging_palette(200, 1, as_cmap=True, center='dark', l=70)
    return cmap


# _______________________________________________________________________
"""
FUNCTIONS
maxconc : returns maximum concentration along a dimension(x,y, or z)
"""


def maxconc(draash, time, enslist, level, dim='y'):
    """
    Returns maximum concetration along a dimension, x, y, or z.
    draash : xarray DataArray
    time   : datetime object
    enslist : list of integers, or integer
    level : list of integers (indices of vertical levels to choose).
            should NOT include the deposition layer if dim='y' or dim='x'.
    """
    source = 0
    # L returns maximum concentration at any level.
    r2 = draash.sel(time=time)
    if 'ens' in draash.dims:
        r2 = r2.isel(ens=enslist)
    if 'source' in draash.dims:
        r2 = r2.isel(source=source)
    r2 = r2.isel(z=level)
    rtot = r2.max(dim=dim)
    # find maximum among all ensemble members as well.
    if 'ens' in draash.dims:
        if len(enslist) > 1:
            rtot = rtot.max(dim='ens')
        else:
            rtot = rtot.isel(ens=0)
    return rtot


def topheight(draash, time, ens, level, thresh=0.01, tp=0):
    source = 0
    if isinstance(level, int):
        level = [level]
    iii = 0
    for lev in level:
        lev_value = draash.z.values[lev]
        r2 = draash.sel(time=time)
        r2 = r2.isel(ens=ens)
        r2 = r2.isel(source=source, z=lev)
        # place zeros where it is below threshold
        r2 = r2.where(r2 >= thresh)
        r2 = r2.fillna(0)
        # place ones where it is above threshold
        r2 = r2.where(r2 < thresh)
        r2 = r2.fillna(lev_value)
        if iii == 0:
            rtot = r2
            rtot.expand_dims('z')
        else:
            r2.expand_dims('z')
            rtot = xr.concat([rtot, r2], 'z')
        rht = rtot.max(dim='z')
    return rht


def choose_and_stack(indra, enslist=None, sourcelist=None):
    """
    Inputs:
    indra : xarray Data Array such as produced by combine_dataset or hysp_massload
    enslist: numpy ndarray, list, or str of values in the 'ens' dimension.
    Returns:
    dra : xarray Data Array.
          "ens" and "source" dimension have been combined into one dimension. "ensemble"
          If "z" coordinate not present it is added.
    """
    dra = indra.copy()

    # if a massloading dataset with no z coordinate is input.
    # then expand dimension to z so can process.
    if 'z' not in dra.coords:
        dra = dra.expand_dims('z')

    if isinstance(enslist, np.ndarray) or isinstance(enslist, list) or isinstance(enslist, str):
        dra = dra.sel(ens=enslist)

    if 'source' in dra.coords and sourcelist:
        dra = dra.sel(source=sourcelist)
        dra = dra.stack(ens=("ens", "source"))
    return dra


def APL(indra, problev=50, enslist=None, sourcelist=None, level=None):
    newra = APLra(indra, enslist, sourcelist, level)
    vpi = np.where(newra.percent_level.values <= problev)
    # if pick value lower than one available then just provide lowest level.
    if not vpi[0]:
        pindex = 0
    else:
        pindex = vpi[0][-1]
    return newra.sel(index=pindex)


def APLra(indra, enslist=None, sourcelist=None, level=None):
    """
    indra: xarray data array. currently must not have dimensions of time.


    Returns:
           xarray data array sorted along the 'ensemble' dimension.
           "ensemble" dimension is replaced by "percent_level" dimension.
    """
    # Applied percentile level
    # currently only work for mass loading.
    dra = indra.copy()
    dra = choose_and_stack(indra, enslist, sourcelist)
    # change this when modifying for concentrations.
    dra2 = dra.isel(z=0)
    # TODO not sure what correct value for transpose_coords should be?
    dra2 = dra2.transpose('ensemble', 'y', 'x', transpose_coords=False)
    coords = dra2.coords
    coords2 = {'latitude': coords['latitude'],
               'longitude': coords['longitude'],
               'x': coords['x'],
               'y': coords['y']}
    dims = dra2.dims
    dvalues = dra2.values.copy()
    # sort along the ensemble axis.
    # sort is an inplace operation. returns an empty array.
    # TODO: check that the ensemble axis is the first one.
    dvalues.sort(axis=0)
    # instead of an 'ensemble' dimension, now have an 'index' dimension.
    newra = xr.DataArray(dvalues, dims=['index', dims[1], dims[2]], coords=coords2)
    percents = 100*(newra.index.values+1) / len(newra.index.values)
    newra = newra.assign_coords(percent_level=('index', percents))
    return newra

def volcATL(indra):
    newra = indra.MER * indra
    return ATL(indra)

def ATL(indra, enslist=None, sourcelist=None,
        thresh=0.2, level=None, norm=False, weights=1):
    """
     Applied Threshold Level (also ensemble frequency of exceedance).

     indra: xr dataarray produced by combine_dataset or by hysp_massload
     enslist : list of values to use fo 'ens' coordinate
     sourcelist : list of values to use for 'source' coordinate
     level : integer or list of integers of vertical levels to use.

     # weights need to add up to 1.

     Returns array with number of ensemble members above
     given threshold at each location.
     norm : boolean
            if True return percentage of members.
            if False return number of members.
     """
    dra = indra.copy()
    dra = choose_and_stack(indra, enslist, sourcelist)
    if isinstance(level, int):
        level = [level]
    if not level:
        mlen = len(dra.z.values)
        level = np.arange(0, mlen)
    # not sure if this loop over the levels is needed.
    for iii, lev in enumerate(level):
        # dra2 = dra.sel(ens=enslist)
        dra2 = dra.isel(z=lev)
        if "source" in dra2.coords:
            dra2 = dra2.isel(source=source)
        # place zeros where it is below threshold
        dra2 = dra2.where(dra2 >= thresh)
        dra2 = dra2.fillna(0)
        # place ones where it is above threshold
        dra2 = dra2.where(dra2 < thresh)
        dra2 = dra2.fillna(1)
        # modify for the weights
        # each level in the 'ens' dimension would
        # need to be multiplied by the individual weight.
        # dra2 = dra2 * weights.
        # ensemble members were above threshold at each location.
        dra2 = dra2.sum(dim=["ens"])
        if iii == 0:
            rtot = dra2
            rtot.expand_dims("z")
        else:
            dra2.expand_dims("z")
            rtot = xr.concat([rtot, dra2], "z")
    # This gives you maximum value that were above concentration
    # at each location.
    if norm:
        nmembers = len(enslist)
        rtot = rtot / nmembers
    return rtot


def ens_mean(draash, time, enslist, level=1):
    """
    draash : xarray
    time : datetime object : date in time coordinate
    enslist: list of integers.
    level : int : index of level in z coordinate.
    """
    source = 0
    sns.set()
    m2 = draash.sel(time=time)
    m2 = m2.isel(ens=enslist, source=source)
    m2 = m2.mean(dim=['ens'])
    if level > 0:
        m2 = m2.isel(z=level)
    return m2


def make_ATL_hysp(xra, variable='p006', threshold=0., MER=None):
    """
    Uses threshold value to make binary field of ensemble members.
    For use in statistics calculations. Based on mass loading, no vertical component
    Inputs:
    xra: xarray dataset - netcdf files from hysplit.combine_dataset
    variable: variable name (string) from xra
    threshold: ash threshold - default=0. (float/integer)
    MER: mass eruption rate - default is Mastin equation MER from xra attributes, units are  (float/integer)
    Output:
    xrabin: binary xarray dataarray (source, x, y)
    """

    xra2 = xra[variable] * 1000.  # Each vertical layer is 1000m - MAY WANT TO ALLOW INPUT
    xra2 = xra2.sum(dim='z')  # Summing along z makes the units g/m^2
    # May want to add loops for time and ensemble dimension in the future
    # MER must used for two ensemble members
    if MER == None:
        MER = xra.attrs['Fine Ash MER']  # Allowing for MER input - default is Mastin equation MER

    xra3 = xra2
    a = 0
    while a < len(xra2.source):
        if xra2[a, :, :].source in ('CylinderSource', 'LineSource'):
            xra3[a, :, :] = xra2[a, :, :] * MER
        else:
            xra3[a, :, :] = xra2[a, :, :]
        a += 1

    xrabin = xr.where(xra3 >= threshold, 1., 0.)
    return xrabin
