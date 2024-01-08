# ----- ------------------------------------------------------------------------
# Air Resources Laboratory
#
# runhelper.py -
#
# Classes
# Helper class contains functions for executing commands.
# JobSetUp class setups the dictionary which contains information for ash runs.
# ConcplotColors class

# Functions
# make_inputs_from_file . returns an instance of JobSetUp class.
# make_dir
# list_dirs

import datetime
import glob
import logging
import os
import pathlib
# import pytz
import shutil
import subprocess
import sys

# 18 APR 2020 (SYZ) - Initial.
# 15 Jun 2020 (AMC) - Adapted from locusts.py
# 07 Jun 2023 (AMC) - move ConcplotColors to utilhysplit.plotutils directory
# 08 Jan 2024 (AMC) - move FileNameComposer to filenamer.py
# -----------------------------------------------------------------------------
from utilhysplit.hcontrol import NameList

logger = logging.getLogger(__name__)


class VolcList:
    def __init__(self, vfile):
        import pandas as pd

        self.df = pd.read_csv(
            vfile,
            header=0,
            index_col=False,
            names=[
                "volcano_name",
                "volcano_region",
                "volcano_lat",
                "volcano_lon",
                "volcano_elevation",
                "type",
            ],
        )
        self.empty = pd.DataFrame()
        self.vlist = self.df.volcano_name.unique()

    def find_exact(self, vname):
        match = [x for x in self.vlist if vname.lower() == x.lower()]
        if match:
            return self.df[self.df.volcano_name == match[0]]
        else:
            return self.empty

    def find_close(self, vname):
        possible = [x for x in self.vlist if vname.lower() in x.lower()]
        return self.df[self.df["volcano_name"].isin(possible)]

    def find_start(self, vname):
        lnn = len(vname)
        possible = [x for x in self.vlist if vname.lower() in x[0:lnn].lower()]
        return self.df[self.df["volcano_name"].isin(possible)]

    def find(self, vname):
        guess = self.find_exact(vname)
        if guess.empty:
            guess = self.find_start(vname)
        if guess.empty:
            guess = self.find_close(vname)
        return guess

    def get_record(self, vname):
        df = self.find(vname)
        return df.to_dict(orient="records")


def complicated2str(a):
    """
    Convert nested list of dictionary and strings to a string
    """
    rstr = ""
    if isinstance(a, list):
        bbb = [complicated2str(x) for x in a]
        bbb = str.join(", ", bbb)
        rstr += bbb
    elif isinstance(a, dict):
        for key in a.keys():
            bbb = str(key) + ":" + complicated2str(a[key]) + "\n"
            rstr += bbb
    else:
        bbb = str(a)
        rstr += bbb
    return rstr


def is_input_complete(ilist, inp):
    """
    ilist : list of tuples (key, 'req' or 'opt')
            The key is the dictionary key that should be present.
            'req' is used if it is required
            'opt' is used if it is optional.

    inp   : dictionary to check against ilist.

    RETURNS
        True if all required values are present
        False if any required values not present

    logger writes warning if required value not present.
    logger writes info if optional values not present
    logger writes info if extra values are present.
    """

    complete = True

    if isinstance(ilist[0], tuple):
        for iii in ilist:
            if iii[0] not in inp.keys():
                if iii[1] == "req":
                    logger.warning("Input does not contain {}".format(iii[0]))
                    complete = False
                else:
                    logger.info("Input does not contain optional {}".format(iii[0]))
    else:
        for iii in ilist:
            if iii not in inp.keys():
                logger.warning("Input does not contain {}".format(iii[0]))
                complete = False

    if isinstance(ilist[0], tuple):
        zlist = list(zip(*ilist))[0]
    else:
        zlist = ilist
    extra = [x for x in inp.keys() if x not in zlist]
    for val in extra:
        logger.debug("Extra inputs {}".format(val))
    return complete


def make_dir(data_dir, newdir="pc_corrected", verbose=False):
    """Create new directory if it does not exist.
    Inputs:
    datadir: Directory in which to create new directory (string)
    newdir: name of new directory (string). If none then will just use datadir.
    """
    if not isinstance(data_dir, str):
        data_dir = str(data_dir)
    if isinstance(newdir, str):
        new_data_dir = os.path.join(data_dir, newdir)
    else:
        new_data_dir = data_dir
    # Go in to given directory, create create new directory if not already there
    if not os.path.exists(new_data_dir):
        orig_umask = os.umask(0)
        os.mkdir(new_data_dir, mode=0o775)
        os.umask(orig_umask)
        if verbose:
            logger.info("Directory created {}".format(new_data_dir))
    if os.path.exists(new_data_dir):
        return True
    else:
        return False


def list_dirs(data_dir):
    """Lists subdirectories within give directory
    Inputs:
    data_dir: directory path of parent directory (string)
    Outputs:
    dirlist: list of subdirectories within data_dir
    """
    # scan directory works with python 3.5 and later.
    dirlist = os.scandir(data_dir)
    newlist = [volc.path.split("/")[-1] for volc in dirlist if volc.is_dir()]
    return sorted(newlist)


class Helper:
    def execute_with_shell(cmd, **kwargs):
        """
        cmd : string
        """
        p = subprocess.Popen(
            " ".join(cmd), shell=True, stdout=sys.stdout, stderr=sys.stderr
        )
        stdoutdata, stderrdata = p.communicate()
        if stdoutdata is not None:
            logger.info(stdoutdata)
            print(stdoutdata)
        else:
            logger.info("executed with no stdout data")
        if stderrdata is not None:
            logger.error(stderrdata)
            print(stderrdata)
        else:
            logger.info("executed with no stderr data")

    def execute(cmd, **kwargs):
        """
        cmd : string
        """
        # p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        p = subprocess.Popen(cmd)
        stdoutdata, stderrdata = p.communicate()
        if stdoutdata is not None:
            logger.info(stdoutdata)
        if stderrdata is not None:
            logger.error(stderrdata)

    def remove(f):
        """
        f : list of strings or string.
        remove file or files in list.
        """
        if isinstance(f, list):
            for g in f:
                if os.path.exists(g):
                    os.remove(g)
        else:
            if os.path.exists(f):
                os.remove(f)

    def move(a, b):
        """
        a : string.
        b : string.
        move file a to b.
        """
        if os.path.exists(a):
            shutil.move(a, b)

    def copy(a, b):
        """
        a : string.
        b : string.
        move file a to b.
        """
        if os.path.exists(a):
            shutil.copy(a, b)

    def move_or_create(a, b):
        if os.path.exists(a):
            shutil.move(a, b)
        else:
            pathlib.Path(b).touch()


def make_inputs_from_file(wdir, config_file="ash_config.txt"):
    jobsetup = JobSetUp()
    # NameList class reads files with configuration key=value
    # into a dictionary.
    config = NameList(fname=config_file, working_directory=wdir)
    config.read()

    if "polygon" in config.nlist.keys():
        config.nlist["polygon"] = process_polygon(config.nlist["polygon"])
        config.nlist["emitfilename"] = "EMITpolygon_{}".format(config.nlist["jobname"])

    # convert dates to datetime objects.
    # temp = list(map(int, config.nlist["start_date"].split(":")))
    if "start_date" not in config.nlist.keys():
        print("cannot find start_date {}".format(config.nlist.keys()))
        sys.exit()
    else:
        try:
            temp = list(map(int, config.nlist["start_date"].split(":")))
        except Exception as eee:
            logger.warning(eee)
            logger.warning(
                "start date not in right format {}".format(config.nlist["start_date"])
            )
            sys.exit()
        config.nlist["start_date"] = datetime.datetime(
            temp[0], temp[1], temp[2], temp[3]
        )

    # convert values to floats where possible.
    for key in config.nlist.keys():
        try:
            val = float(config.nlist[key])
        except:
            val = config.nlist[key]
        # get rid of any white spaces in strings
        config.nlist[key] = val

    jobsetup.inp = config.nlist
    jobsetup.add_plotting_options(config.nlist)
    # puts in defaults if not in the config.nlist
    jobsetup.add_optional_params(config.nlist)
    return jobsetup


class EventSetUp:
    def __init__(self):
        self.inp = {}
        self.set_directories()

    def set_directories(self):
        self.inp["JPSS_DIR"] = "/pub/jpsss_upload"
        self.inp["VOLCAT_LOGFILES"] = "/pub/ECMWF/JPSS/VOLCAT/LogFiles/"
        self.inp["VOLCAT_DIR"] = "/pub/ECMWF/JPSS/VOLCAT/Files/"


def process_polygon(pstring):
    """
    pstring : string representing vertices of polygon.
    The string is like the strings used in volcanic ash advisories.
    N,S are used in front of the latitude points.
    W,E are used in front of the longitude points.
    (latitude, longitude) pairs are separated by dashes -
    latitude, longitude are given without decimal points to the 100th's place.
    Example
    N4843 W10929-N4838 W10834-N4423 W10644-N4413 W10737-N4843 W10929
    Representing
    (48.43,-109.29),(48.38,-108.34),(44.23,-106.44),(44.13,-107.37)
    """

    temp = pstring.split("-")
    lon = -999.0
    lat = -999.0
    vlist = []
    for vertice in temp:
        tvv = vertice.split()
        for val in tvv:
            if "n" in val.lower():
                lat = val.lower()
                lat = lat.replace("n", "")
                lat = int(lat) / 100.0
            elif "s" in val.lower():
                lat = val.lower()
                lat = lat.replace("s", "")
                lat = -1 * int(lat) / 100.0
            elif "w" in val.lower():
                lon = val.lower()
                lon = lon.replace("w", "")
                lon = -1 * int(lon) / 100.0
            elif "e" in val.lower():
                lon = val.lower()
                lon = lon.replace("e", "")
                lon = int(lon) / 100.0
        vlist.append((lon, lat))
    return vlist


class JobSetUp:
    def __init__(self):
        self.inp = {}

    def write(self, fname, wdir="./"):
        inp = self.inp.copy()
        inp["start_date"] = inp["start_date"].strftime("%Y:%m:%d:%H")
        output = NameList(fname=fname, working_directory=wdir)
        output.add_n(inp)
        output.line_ending = ""
        output.write(overwrite=True, noheader=True)

    def parse_inputs(self, a):
        self.add_run_params(a)
        self.add_directories(a)
        self.add_plotting_options(a)
        return self.inp

    def add_input(self, inp, astr, default):
        """
        inp : dictionary with keys to add to self.inp
        astr : key to look for in inp
        default : value to use if key not found in inp.
        """
        if astr in inp.keys():
            self.inp[astr] = inp[astr]
        else:
            self.inp[astr] = default

    def not_used(self, inp):
        self.inp["wetflag"] = inp["usingWetDeposition"]
        self.inp["forwardflag"] = inp["forwardCalculation"]
        # 0 for large down to 3 for small.
        # self.inp['eflag'] = inp['eruptionSize']
        self.inp["polygon"] = None  # input polygon points to start from.

    def add_optional_params(self, inp=None):
        # area emission is used for inverse modeling.
        self.add_input(inp, "area", default=1)
        # self.add_input(inp,'rate',default=1)

    def add_inverse_params(self, inp=None):
        # time resolution for each inverse modeling run.
        # in hours. default 1 hour.
        if not inp:
            inp = {}
        # self.add_input(inp,'timeres',0.5)
        # self.add_input(inp,'rate',default=2)
        self.add_input(inp, "timeres", 1)
        self.add_input(inp, "rate", default=1)
        # vertical resolution for each inverse modeling run.
        # in m. default 1000m.
        self.add_input(inp, "inv_vertical_resolution", 1000)
        # self.add_input(inp,'inv_vertical_resolution',500)

    def add_run_params(self, inp):
        # inputs from web form.
        self.inp["owner"] = inp["owner"]  # TODO fix value
        self.inp["VolcanoName"] = inp["volcano"]["name"]
        self.inp["VolcanoLocation"] = inp["volcano"]["location"]
        self.inp["VolcanoName"] = inp["volcano"]["name"]
        self.inp["latitude"] = float(inp["sourceLatitude"])
        self.inp["longitude"] = float(inp["sourceLongitude"])
        self.inp["bottom"] = float(inp["volcano"]["height"])
        self.inp["top"] = float(inp["plumeHeight"])
        self.inp["durationOfSimulation"] = inp["simulationDurationInHours"]
        self.inp["emissionHours"] = inp["emissionHours"]
        # Must be in UTC.
        self.inp["start_date"] = datetime.datetime(
            inp["startDate"][0],
            inp["startDate"][1],
            inp["startDate"][2],
            inp["startTime"][0],
            inp["startTime"][1],
        )
        self.inp["samplingIntervalHours"] = inp["samplingIntervalInHours"]
        self.inp["meteorologicalData"] = inp["meteorologicalData"]
        # runflag can be 'trajectory' or 'dispersion'
        if inp["runType"] == 0:
            self.inp["runflag"] = "trajectory"
        else:
            self.inp["runflag"] = "dispersion"
        self.inp["jobname"] = inp["runIdentifier"]
        if "PolygonString" in inp.keys():
            self.inp["polygon"] = inp[
                "PolygonString"
            ]  # input polygon points to start from.

        # eflag should be a float. Large values give smaller plumes.
        # concentration value is multiplied by 10^(-1 * eflag)
        # This is because currently VAAC is used to using an 'ash reduction'
        # of 1 2 or 3 with 3 reducing the ash the most.
        # However large number to reduce plume is somewhat counter-intuitive.
        self.inp["eflag"] = float(inp["eruptionSize"])
        # below are currently always the same.
        self.inp["source_type"] = "uniform"

        # polygon optional input
        if "polygon" in self.inp.keys():
            self.inp["polygon"] = process_polygon(self.inp["polygon"])
            self.inp["emitfilename"] = "EMITpolygon_{}".format(self.inp["jobname"])

    def ensure_trailing_slash(self, dir):
        if dir[-1] != os.sep:
            return dir + os.sep
        return dir

    def add_test_directories(self):
        tdir = "/hysplit-users/alicec/"
        self.inp["HYSPLIT_DIR"] = "{}hdev/".format(tdir)
        self.inp["MAP_DIR"] = "/hysplit-users/alicec/tags/hysplit.v5.3.0/graphics/"
        self.inp["WORK_DIR"] = "/hysplit-users/alicec/tmp/testing/"
        self.inp["DATA_DIR"] = "/hysplit-users/alicec/utilhysplit/ashapp/"
        self.inp["FILES_DIR"] = "./"
        self.inp[
            "PYTHON_EXE"
        ] = "/hysplit-users/alicec/anaconda3/envs/hysplit/bin/python"
        self.inp["forecastDirectory"] = "/pub/forecast/"
        self.inp["archivesDirectory"] = "/pub/archives/"
        self.inp["CONVERT_EXE"] = "convert"
        self.inp["GHOSTSCRIPT_EXE"] = "gs"

    def add_directories(self, inp):
        self.inp["HYSPLIT_DIR"] = inp["readyProperties"]["directory"]["hysplit"]
        self.inp["MAP_DIR"] = inp["readyProperties"]["directory"]["map"]
        self.inp["WORK_DIR"] = self.ensure_trailing_slash(inp["workingDirectory"])
        self.inp["DATA_DIR"] = inp["dataDirectory"]

        ## TO DO CONTROL.default kept with repository in scripts directory.
        ## What will be the full path?
        self.inp["FILES_DIR"] = "./"

        self.inp["CONVERT_EXE"] = inp["readyProperties"]["executable"]["convert"]
        self.inp["GHOSTSCRIPT_EXE"] = inp["readyProperties"]["executable"][
            "ghostscript"
        ]
        self.inp["PYTHON_EXE"] = inp["readyProperties"]["executable"]["python"]

        self.inp["forecastDirectory"] = inp["readyProperties"]["directory"]["forecast"]
        self.inp["archivesDirectory"] = inp["readyProperties"]["directory"]["archives"]

    def add_plotting_options(self, inp):
        self.add_input(inp, "gisOption", 3)
        self.add_input(inp, "zoomFactor", 50)
        self.add_input(inp, "generatingPostscript", True)
        self.inp["generatingPDF"] = True
        self.inp["mapBackground"] = "arlmap"
        self.inp["mapProjection"] = 0
        self.inp["spatialPlotRadius"] = 500.0  # km
        self.inp["generatingPDF"] = True
        self.inp["graphicsResolution"] = 200
        self.inp["zip_compression_level"] = 3

    def make_test_inputs(self):
        vname = "Kilauea"
        # vname = "bezy"
        # vname = 'Reventador'
        self.inp["owner"] = "A. Person"
        self.inp["top"] = 20000
        self.inp["durationOfSimulation"] = 24
        self.inp["rate"] = 1
        self.inp["area"] = 1
        testdate = datetime.datetime.now() - datetime.timedelta(hours=24)
        # testdate = datetime.datetime(2020,10,10,11)
        testminutes = 15
        self.inp["start_date"] = datetime.datetime(
            testdate.year, testdate.month, testdate.day, testdate.hour, testminutes
        )
        if vname == "inverse":
            self.inp["durationOfSimulation"] = 12
            self.inp["top"] = 10000
            testminutes = 0
            self.inp["start_date"] = datetime.datetime(
                testdate.year, testdate.month, testdate.day, testdate.hour, testminutes
            )
            vname = "douglas"

        self.inp["emissionHours"] = 4
        self.inp["meteorologicalData"] = "GFS0p25"
        self.inp["EruptionSize"] = 0
        if vname.lower() == "bezy":
            # bezy data starts at 10/21 at 20:40
            #      ends at 10/22 at 21:10
            self.inp["meteorologicalData"] = "GFS0p25"
            self.inp["VolcanoName"] = "Bezymianny"
            self.inp["start_date"] = datetime.datetime(2020, 10, 21, 19)
            self.inp["durationOfSimulation"] = 36
            self.inp["emissionHours"] = 24
            self.inp["top"] = 15000
            self.inp["bottom"] = 9455
            self.inp["latitude"] = 55.978
            self.inp["longitude"] = 160.587

        if vname.lower() == "raikoke":
            self.inp["meteorologicalData"] = "GFS0p25"
            self.inp["VolcanoName"] = "Raikoke"
            self.inp["start_date"] = datetime.datetime(2019, 6, 21, 18)
            self.inp["durationOfSimulation"] = 24
            self.inp["emissionHours"] = 12
            self.inp["top"] = 15000
            self.inp["bottom"] = 1000
            self.inp["latitude"] = 48.292
            self.inp["longitude"] = 153.25

        if vname.lower() == "douglas":
            self.inp["meteorologicalData"] = "NAMHAK"
            self.inp["VolcanoName"] = "Douglas"
            self.inp["latitude"] = 58.855
            self.inp["longitude"] = -153.54
            self.inp["bottom"] = 7021
        if vname.lower() == "kilauea":
            # self.inp["meteorologicalData"] = "NAMHHI"
            self.inp["meteorologicalData"] = "GFS0p25"
            self.inp["VolcanoName"] = "Kilauea"
            self.inp["latitude"] = 19.421
            self.inp["longitude"] = -155.28
            self.inp["bottom"] = 4009
        if vname == "Reventador":
            self.inp["VolcanoName"] = "Reventador"
            self.inp["latitude"] = -0.08
            self.inp["longitude"] = -77.66
            self.inp["bottom"] = 3562
        if vname == "Veni":
            self.inp["VolcanoName"] = "Veniaminof"
            self.inp["latitude"] = 56.17
            self.inp["longitude"] = -159.38
            self.inp["bottom"] = 2507
        self.inp["samplingIntervalHours"] = 3
        self.inp["eflag"] = 0
        self.add_test_directories()
        self.add_plotting_options(inp={})
        self.inp["source_type"] = "uniform"
        self.inp["jobname"] = "ashtest"
        self.inp["runflag"] = "dispersion"
        return self.inp


# volcat_events.py Events class has
# ndir, pdir, edir, idir and a get_dir method.


class VolcatDirectories:
    ilist = ["JPSS_DIR", "VOLCAT_LOGFILES", "VOLCAT_DIR"]

    def __init__(self, inp, volcano_name):
        self.get_dir(inp)

    def defaults(self):
        self.inp["JPSS_DIR"] = "/pub/jpsss_upload"
        self.inp["VOLCAT_LOGFILES"] = "/pub/ECMWF/JPSS/VOLCAT/LogFiles/"
        self.inp["VOLCAT_DIR"] = "/pub/ECMWF/JPSS/VOLCAT/Files/"
        return -1

    def event_directories(self, inp, volcano_name, verbose=False, make=True):
        """
        inp : dictionary with key VOLCAT_DIR

        ndir : top level directory where volcat netcdf files are located.
        pdir : directory where the parallax corrected files are to be
        edir : directory where emit-times files are written.
        idir : directory where runs for inversions are written.

        """
        if not isinstance(inp, dict):
            return None
        if "VOLCAT_DIR" not in inp.keys():
            logger.warning("get_dir method input does not contain VOLCAT_DIR")
            return None
        #tdir = inp["VOLCAT_DIR"]
        if volcano_name != "Unknown":
            ndir = os.path.join(inp["VOLCAT_DIR"], self.volcano_name)
        else:
            ndir = inp["VOLCAT_DIR"]
        pdir = os.path.join(ndir, "pc_corrected")
        edir = os.path.join(ndir, "emitimes")
        idir = os.path.join(ndir, "inverse")

        if verbose:
            logger.info("Downloading to {}".format(ndir))
            logger.info("parallax corrected to {}".format(pdir))
            logger.info("emit times files to {}".format(edir))
        # set_dir(ndir, pdir, edir, idir, make=make)
        return ndir, pdir, edir, idir




