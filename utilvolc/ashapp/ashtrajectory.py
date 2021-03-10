#!/opt/Tools/anaconda3/envs/hysplit/bin/python
# -----------------------------------------------------------------------------
# Air Resources Laboratory
#
# ashtrajectory.py - run HYSPLIT model on web and create plots
#
# 01 JUN 2020 (AMC) - adapted from locusts-run.py
# -----------------------------------------------------------------------------
# To run in offline mode use python ash_run.py -777
# -----------------------------------------------------------------------------

import datetime
import logging
import os
from runhelper import Helper 
from ashbase import AshRun

logger = logging.getLogger(__name__)

def print_usage():
    print("""\
USAGE: TrajectoryAshRun 
""")

# Base class is AshRun
class TrajectoryAshRun(AshRun):
    def __init__(self, JOBID):
        super().__init__(JOBID)

    def additional_control_setup(self, control, stage=0):
        """
        control : HycsControl object
        stage : int or str
        """
        # for dispersion control
        # add location of eruption
        # trajectories every 2 km 
        vent = self.inp['bottom']
        height = self.inp['top']
        lat = self.inp['latitude']
        lon = self.inp['longitude']
        control.remove_locations()
        plist = list(range(0,26000,2000))
        #height_list = [x for x in plist if x > vent and x < height]
        # This returns list of FL every FL50. 12 levels total.
        height_list,rlist = self.set_levels_A()
        # For trajectory every FL100 is sufficient
        height_list = height_list[::2]        
        if not height_list:
            logger.info('No height found between vent {} and top height'
                        ' {}. Using Default heights'.format(vent, height))
            height_list = list(range(5000,25000,5000))

        [control.add_location((lat,lon), ht) for ht in height_list] 
        control.outdir = self.inp['WORK_DIR']
        control.outfile = self.filelocator.get_tdump_filename(stage)        

    def run_model(self):
        # make control and setup files
        self.compose_control(stage=0, rtype='trajectory')
        self.compose_setup(stage=0)
        # start run and wait for it to finish..
        c = [os.path.join(self.inp['HYSPLIT_DIR'], 'exec', 'hyts_std'), self.JOBID]
        logger.info('Running {} with job id {}'.format('hyts_std', c[1]))
        Helper.execute(c)

    def create_plots(self, redraw=False, stage=0):
        logger.info('creating graphics')
        self.create_trajectory_plot(stage)


    def make_awips_netcdf(self):
        """
        TO DO : cannot currently generate AWIPS2 formatted files for trajectories.
        """
        return None


    def after_run_check(self, update=False):
        # Check for the tdump/cdump file
        rval = True
        fn = self.filelocator.get_tdump_filename(stage=0)
        logger.debug("Looking for tdump file " + fn)
        if not os.path.exists(fn):
            rval = False
            if update:
                logger.error(
                    "******************************************************************************"
                )
                logger.error(
                    "The model has crashed. Check the HYSPLIT Message file for further information."
                )
                logger.error(
                    "******************************************************************************"
                )
        return rval

    def create_trajectory_plot(self,stage):
        logger.info('Creating trajectory graphics for job {}.'.format(self.JOBID))

        ptype = 'png'
        outputname = self.filelocator.get_trajplot_filename(stage, ptype=ptype)
        gisopt = 3 #self.inp.gisOption
        if gisopt == 1:
            msg = 'GIS shapefile creation for job {}.'.format(self.JOBID)
            logger.info(msg)
            fn = self.filelocator.get_gis_status_filename()
            with open(fn, 'at') as f:
                f.write(msg)
                f.write('\n')
        elif gisopt == 3:
            msg = 'Google Earth file creation for job {}.'.format(self.JOBID)
            logger.info(msg)
            fn = self.filelocator.get_gis_status_filename()
            with open(fn, 'at') as f:
                f.write(msg)
                f.write('\n')

        mapopt = self.inp['mapBackground']
        if mapopt == 'terrain':
            maparg = '--street-map=0'
        elif mapopt == 'toner':
            maparg = '--street-map=1'
        else:
            maparg = '-j' + os.path.join(self.inp['MAP_DIR'], 'arlmap')

        # trajectory output
        fns = [self.filelocator.get_tdump_filename(stage=stage)]
        c = [self.inp['PYTHON_EXE'],
             os.path.join(self.inp['HYSPLIT_DIR'], 'exec', 'trajplot.py'),
             '-i' + '+'.join(fns),
             '+n',
             '-o' + outputname,
             '-s1',
             '-l1',
             maparg,
             '-a{}'.format(gisopt),
             '-k1'
             #'-g0:{}'.format(self.inp['spatialPlotRadius']),
             #'-p' + fileidentifier,
             #'-m{}'.format(self.inp['mapProjection']),
             #'-z{}'.format(self.inp['zoomFactor']),
             ]

        logger.debug('Trajplot {}'.format(' '.join(c)))
        Helper.execute(c)
      
        # add frame number to filename 
        outputname =\
             self.filelocator.get_trajplot_filename(stage,ptype=ptype,frame=1)
        logger.debug('Looking for-------{}'.format(outputname))
        if not os.path.exists(outputname):
            logger.error('******************************************************************************')
            logger.error('The model was not able to create a graphic file for job {}.'.format(self.JOBID))
            logger.error('******************************************************************************')
            self.update_run_status(self.JOBID, 'GRAPHICS_FAILED')
            logger.info(datetime.datetime.now())
        else:
            kmlname = self.filelocator.get_trajplot_kml_filename(stage,frame=1)
            self.generate_kmz([kmlname],self.filelocator.get_kmz_filename(stage='traj'))



