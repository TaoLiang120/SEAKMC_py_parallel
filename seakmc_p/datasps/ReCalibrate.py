import copy
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

import seakmc_p.mpiconf.MPIconf as mympi
from seakmc_p.input.Input import SP_COMPACT_HEADER4Delete
from seakmc_p.mpiconf.error_exit import error_exit

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def recalib_energy_single(itask_start, thiscolor, comm_split,
                          idtasks, thissett, DataSPs, seakmcdata, AVitags, Eground, ReBias, object_dict):
    force_evaluator = object_dict['force_evaluator']
    rank_local = comm_split.Get_rank()

    lidt = itask_start + thiscolor
    thisid = idtasks[lidt]
    iav = DataSPs.localiav[thisid]
    idav = DataSPs.df_SPs[iav].at[0, "idav"]
    lisp = DataSPs.localisp[thisid]

    thisRecal = True
    if isinstance(thissett.saddle_point["Thres4Recalib"], float) or isinstance(thissett.saddle_point["Thres4Recalib"],
                                                                               int):
        if DataSPs.df_SPs[iav].at[lisp, "barrier"] - DataSPs.barriermin > thissett.saddle_point[
            "Thres4Recalib"]: thisRecal = False

    thisidens = {"isp": thisid, "barr": None, "ebias": None}
    thisid4del = None
    thisreason = None
    if thisRecal:
        thisdata = copy.deepcopy(seakmcdata)
        thisdata.update_coords_from_disps(idav, DataSPs.disps[thisid], AVitags)
        [thisesp, coords, isValid, errormsg] = force_evaluator.run_runner("MD0", thisdata, thiscolor,
                                                                          nactive=thisdata.natoms, thisExports=None,
                                                                          comm=comm_split)
        if not isValid:
            error_exit(errormsg)
        thisbarr = round(thisesp - Eground, thissett.system["float_precision"])
        thisidens["barr"] = thisbarr
        if ReBias:
            thisdata = copy.deepcopy(seakmcdata)
            thisdata.update_coords_from_disps(idav, DataSPs.fdisps[thisid], AVitags)
            [thisefin, coords, isValid, errormsg] = force_evaluator.run_runner("MD0", thisdata, thiscolor,
                                                                               nactive=thisdata.natoms,
                                                                               thisExports=None, comm=comm_split)
            if not isValid:
                error_exit(errormsg)
            thisebias = round(thisefin - Eground, thissett.system["float_precision"])
            thisidens["ebias"] = thisebias
        if thisbarr < thissett.saddle_point["BarrierMin"]:
            thisid4del = thisid
            thisreason = "minB_Re"
        elif ReBias:
            if thisbarr - thisebias < thissett.saddle_point["BarrierMin"]:
                thisid4del = thisid
                thisreason = "minBB_Re"
    thisdata = None
    return thisidens, thisid4del, thisreason


def recalibrate_energy(thissett, DataSPs, seakmcdata, AVitags, Eground, object_dict):
    ntask_tot = DataSPs.nSP
    nproc_task = thissett.force_evaluator['nproc4ReCal']
    if nproc_task > size_world:
        errormsg = "Number of processors is smaller than number of processors for CalBarrsInData"
        error_exit(errormsg)

    start_proc = 0
    ntask_time = mympi.get_ntask_time(nproc_task, start_proc=start_proc, thiscomm=None)
    comm_split, thiscolor = mympi.split_communicator(nproc_task, start_proc=start_proc, thiscomm=None)

    LogWriter = object_dict['LogWriter']
    ReBias = thissett.saddle_point["CalEbiasInData"]
    idtasks = np.arange(DataSPs.nSP)
    if rank_world == 0:
        tic = time.time()
        idens = np.array([], dtype=int)
        ids4del = np.array([], dtype=int)
        reasons = np.array([], dtype=str)
        #typemask = np.zeros(ntask_tot, dtype=bool)
        logstr = "\n" + "ReCalibrating the energy with data ..."
        LogWriter.write_data(logstr)

    ntask_left = ntask_tot
    ntask_time = min(ntask_time, ntask_left)
    itask_start = 0
    while ntask_left > 0:
        if thiscolor < ntask_time:
            thisidens, thisid4del, thisreason = recalib_energy_single(itask_start, thiscolor, comm_split,
                                                                      idtasks, thissett, DataSPs, seakmcdata, AVitags,
                                                                      Eground, ReBias, object_dict)
        else:
            thisidens = None
            thisid4del = None
            thisreason = None

        if rank_world == 0:
            idens = np.append(idens, [thisidens])
            ids4del = np.append(ids4del, [thisid4del])
            reasons = np.append(reasons, [thisreason])

            for i in range(1, ntask_time):
                thisidens = comm_world.recv(source=i * nproc_task, tag=i * 10 + 1)
                thisid4del = comm_world.recv(source=i * nproc_task, tag=i * 10 + 2)
                thisreason = comm_world.recv(source=i * nproc_task, tag=i * 10 + 3)
                idens = np.append(idens, [thisidens])
                ids4del = np.append(ids4del, [thisid4del])
                reasons = np.append(reasons, [thisreason])
        else:
            if thiscolor > 0 and thiscolor < ntask_time and comm_split.Get_rank() == 0:
                comm_world.send(thisidens, dest=0, tag=thiscolor * 10 + 1)
                comm_world.send(thisid4del, dest=0, tag=thiscolor * 10 + 2)
                comm_world.send(thisreason, dest=0, tag=thiscolor * 10 + 3)
            else:
                pass

        comm_world.Barrier()

        itask_start += ntask_time
        ntask_left = ntask_left - ntask_time
        ntask_time = min(ntask_time, ntask_left)

    comm_world.Barrier()

    df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
    to_delete_localisp = np.array([], dtype=int)
    if rank_world == 0:
        for i in range(len(idens)):
            thisidens = idens[i]
            if thisidens is not None:
                isp = thisidens["isp"]
                iav = DataSPs.localiav[isp]
                lisp = DataSPs.localisp[isp]
                if thisidens["barr"] is not None:
                    if isinstance(thisidens["barr"], float):
                        DataSPs.df_SPs[iav].at[lisp, "barrier"] = thisidens["barr"]
                if thisidens["ebias"] is not None:
                    if isinstance(thisidens["ebias"], float):
                        DataSPs.df_SPs[iav].at[lisp, "ebias"] = thisidens["ebias"]

        for i in range(len(ids4del)):
            isp = ids4del[i]
            thisreason = reasons[i]
            if isp is not None:
                to_delete_localisp = np.append(to_delete_localisp, [isp])
                iav = DataSPs.localiav[isp]
                lisp = DataSPs.localisp[isp]
                thisrow = DataSPs.df_SPs[iav].loc[lisp].to_dict()
                thisrow["reason"] = thisreason
                df_delete_this.loc[len(df_delete_this)] = thisrow

        DataSPs.reorganization(to_delete_localisp)
        toc = time.time()
        logstr = f"Finished recalibration and time cost is {round(toc - tic, 2)}s."
        logstr += "\n" + "KMC_istep_SPs.csv has energies before the recalibration."
        logstr += "\n" + "KMC_istep_Prob.csv has energies after the recalibration."
        logstr += "\n" + "-----------------------------------------------------------------"
        LogWriter.write_data(logstr)
    else:
        DataSPs = None
        df_delete_this = None

    DataSPs = comm_world.bcast(DataSPs, root=0)
    df_delete_this = comm_world.bcast(df_delete_this, root=0)

    comm_split.Free()
    comm_world.Barrier()

    return DataSPs, df_delete_this
