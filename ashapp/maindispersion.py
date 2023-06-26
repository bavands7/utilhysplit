#!/opt/Tools/anaconda3/envs/hysplit/bin/python
# -----------------------------------------------------------------------------
# Air Resources Laboratory
#
# ash_run.py - run HYSPLIT model on web and create plots
#
# 01 JUN 2020 (AMC) - adapted from locusts-run.py
# 09 DEC 2022 (AMC) - changed latspan to 180 and lonspan to 360 for global grid.
# 09 DEC 2022 (AMC) - changed numpar to 10000. this should not be hardwired.
# 28 FEB 2023 (AMC) - change to write_cxra so if netcdf file exists then set it as self.cxra
# 28 FEB 2023 (AMC) - check attributes added as stand alone function but still kept as static method.
#
# -----------------------------------------------------------------------------
# To run in offline mode use python ash_run.py -999
#
#
# -----------------------------------------------------------------------------

import datetime
import logging
import os
import shutil

import requests

from ashapp.ashruninterface import MainRunInterface
from ashapp.rundispersion import RunDispersion
from ashapp.ensembledispersion import EnsembleDispersion
from ashapp.graphicsdispersion import GraphicsDispersion
from ashapp.outputdispersion import OutputDispersion
from ashapp import utils

# from ashapp import  utils
# from ashapp.ashnetcdf import HYSPLITAshNetcdf

# from utilvolc.volcMER import HT2unit

logger = logging.getLogger(__name__)


class MainDispersion(MainRunInterface):
    def __init__(self, inp, JOBID):
        # 14 instance attributes
        self.JOBID = JOBID  # string

        self.ilist = [
            "MAP_DIR",
            "WORK_DIR",
            "CONVERT_EXE",
            "GHOSTSCRIPT_EXE",
            "PYTHON_EXE",
        ]

        inp["jobid"] = JOBID
        self.inp = inp  # dictionary from JobSetUP
        self.apistr = None
        self.urlstr = None
        self.headerstr = None

        self.filelocator = None
        self.maptexthash = {}
        self.awips = True

        self._modelrun = RunDispersion(inp)
        self._modeloutput = OutputDispersion(inp, [])
        self._modelgraphics = GraphicsDispersion(inp)

        utils.setup_logger()

    @property
    def JOBID(self):
        return self._JOBID

    @JOBID.setter
    def JOBID(self, JOBID):
        self._JOBID = str(JOBID)

    @property
    def inp(self):
        return self._inp

    @inp.setter
    def inp(self, inp):
        complete = True
        for iii in self.ilist:
            if iii not in inp.keys():
                logger.warning("Input does not contain {}".format(iii))
                complete = False
        if complete:
            logger.info("Input contains all fields")
        self._inp = inp

    @property
    def modelrun(self):
        return self._modelrun

    @modelrun.setter
    def modelrun(self, mrun):
        self._modelrun = mrun

    @property
    def modeloutput(self):
        return self._modeloutput

    @property
    def modelgraphics(self):
        return self._modelgraphics

    def add_api_info(self, apistr, urlstr, headerstr):
        self.apistr = apistr
        self.urlstr = urlstr
        self.headerstr = headerstr

    def update_run_status(self, jobId, status):
        if self.apistr:
            API_KEY = os.environ[self.apistr]
            RUN_URL = os.environ[self.urlstr]
            statusUrl = "{}/status/{}/{}".format(RUN_URL, jobId, status)
            r = requests.put(statusUrl, headers={self.headerstr: API_KEY})
        else:
            logger.info("Running in offline test mode")
        logger.info("Posted status {} for job {}".format(status, jobId))

    def handle_crash(self, stage=0):
        self.update_run_status(self.JOBID, "CRASHED")
        logger.info("The model has crashed for job {}.".format(self.JOBID))
        logger.info(datetime.datetime.now())

    def debug_message(self):
        # debug messages
        logger.debug("HYSPLIT_DIR     = {}".format(self.inp["HYSPLIT_DIR"]))
        logger.debug("MAP_DIR         = {}".format(self.inp["MAP_DIR"]))
        logger.debug("WORK_DIR        = {}".format(self.inp["WORK_DIR"]))
        logger.debug("CONVERT_EXE     = {}".format(self.inp["CONVERT_EXE"]))
        logger.debug("GHOSTSCRIPT_EXE = {}".format(self.inp["GHOSTSCRIPT_EXE"]))
        logger.debug("PYTHON_EXE      = {}".format(self.inp["PYTHON_EXE"]))

    def doit(self):
        """
        Main work flow.
        """
        self.debug_message()
        os.chdir(self.inp["WORK_DIR"])
        if not os.path.exists("ASCDATA.CFG"):
            shutil.copyfile(
                self.inp["HYSPLIT_DIR"] + "/bdyfiles/ASCDATA.CFG", "ASCDATA.CFG"
            )
        logger.info("Please wait for further information....")
        logger.info("Model submitted on {}".format(datetime.datetime.now()))

        # run_model will check if the run has already been done.
        # self.modelrun.run_model(overwrite=False)
        self.modelrun.run(overwrite=False)

        # make the model output.
        self.modeloutput.inputlist = self.modelrun.filelist
        self.modeloutput.postprocess()

        # make the graphics
        if self.modeloutput.check():
            self.modelgraphics.inputlist = self.modeloutput.outputlist
            self.modelgraphics.postprocess()

        # update the run status
        self.update_run_status(self.JOBID, "COMPLETED")

        # cleanup files
        self.cleanup()

    def cleanup(self):
        return True

    def after_run_check(self, stage=0, update=False):
        return True
        # Check for the tdump/cdump file
        # should this be done here?
        # rval = True
        # fnlist = [x for x in self.modelrun.filelist if 'cdump' in x]
        # for fn in fnlist:
        #    if not os.path.exists(fn):
        #        rval = False
        #        logger.info("NOT found cdump file " + fn)
        #    else:
        #        logger.info("found cdump file " + fn[0])

        # if update and not rval:
        #    logger.error(
        #        "******************************************************************************"
        #    )
        #    logger.error(
        #        "The model has crashed. Check the HYSPLIT Message file for further information."
        #    )
        #    logger.error(
        #        "******************************************************************************"
        #    )
        #    self.handle_crash(stage=0)
        # return rval


class MainEnsemble(MainDispersion):
    def __init__(self, inp, JOBID):
        """
        modelrun attribute is the EnsembleDispersion class.
        """


        self.JOBID = JOBID  # string

        self.ilist = [
            "MAP_DIR",
            "WORK_DIR",
            "CONVERT_EXE",
            "GHOSTSCRIPT_EXE",
            "PYTHON_EXE",
        ]

        inp["jobid"] = JOBID
        self.inp = inp  # dictionary from JobSetUP
        self.apistr = None
        self.urlstr = None
        self.headerstr = None

        self.filelocator = None
        self.maptexthash = {}
        self.awips = True

        self._modelrun = EnsembleDispersion(inp, self.JOBID)
        self._modeloutput = OutputDispersion(inp, [])
        self._modelgraphics = GraphicsDispersion(inp)

        utils.setup_logger()
