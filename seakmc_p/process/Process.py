import os
import shutil
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

import seakmc_p.datasps.DataKMC as dataKMC
import seakmc_p.datasps.DataSPS as dataSPS
import seakmc_p.datasps.PreSPS as preSPS
import seakmc_p.datasps.ReCalibrate as myRecal
import seakmc_p.general.DataOut as dataout
import seakmc_p.process.DataDyn as mydatadyn
from seakmc_p.core.data import SeakmcData
from seakmc_p.input.Input import SP_COMPACT_HEADER4Delete, SP_DATA_HEADER
from seakmc_p.kmc.KMC import SuperBasin
from seakmc_p.restart.Restart import RESTART
from seakmc_p.spsearch.SaddlePoints import Data_SPs
from seakmc_p.mpiconf.error_exit import error_exit

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def run_seakmc(thissett, seakmcdata, object_dict, Eground, thisRestart):
    out_paths = object_dict['out_paths']
    force_evaluator = object_dict['force_evaluator']
    LogWriter = object_dict['LogWriter']
    thisSummary = object_dict['thisSummary']
    DFWriter = object_dict['DFWriter']

    THIS_PATH = out_paths[-1]
    thisExports = thisSummary.export_dict

    if thisRestart is None:
        simulation_time = 0.0
        thisSuperBasin = SuperBasin([], thissett.kinetic_MC["Temp"])
        thisSuperBasin.initialization()
        DefectBank_list = []
        if thissett.defect_bank["LoadDB"]:
            DefectBank_list = preSPS.load_DefectBanks(thissett.defect_bank, out_paths[2],
                                                      significant_figures=thissett.system["significant_figures"])
        istep_this = 0
    else:
        thisSuperBasin = thisRestart.thisSuperBasin
        DefectBank_list = thisRestart.DefectBank_list
        istep_this = thisRestart.istep_this
        simulation_time = thisRestart.simulation_time
        if thissett.system["Restart"]["Reset_Simulation_Time"]: simulation_time = 0.0

    if isinstance(thissett.active_volume["DefectCenter4RT_SetMolID"], list):
        last_de_center = thissett.active_volume["DefectCenter4RT_SetMolID"]
    else:
        last_de_center = None

    comm_world.Barrier()

    for istep in range(istep_this, thissett.kinetic_MC['NSteps']):
        if rank_world == 0:
            tickmc = time.time()
            DFWriter.init_deleted_SPs(istep)
            DFWriter.init_SPs(istep)
            logstr = f"istep KMC: {istep}"
            LogWriter.write_data(logstr)

        if thisRestart is None:
            seakmcdata.get_defects(LogWriter, last_de_center=last_de_center)
            dataout.visualize_data_AVs(thissett.visual, seakmcdata, istep, out_paths[1])
            emptya = np.array([], dtype=int)
            AVitags = [emptya for i in range(seakmcdata.ndefects)]
            DataSPs = Data_SPs(istep, seakmcdata.ndefects)
            DataSPs.initialization()
            df_delete_SPs = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
            undo_idavs = np.arange(seakmcdata.ndefects, dtype=int)
            finished_AVs = 0
            if rank_world == 0:
                logstr = (f"The ground energy is "
                          f"{round(Eground, thissett.system['float_precision'])} eV at {istep} KMC step!")
                logstr += ("\n" +
                           f"There are {seakmcdata.ndefects} defects (active volumes) in data at {istep} KMC step!")
                logstr += ("\n" +
                           f"The fractional coords of the defect center are "
                           f"{np.around(seakmcdata.de_center, decimals=thissett.system['float_precision'])} "
                           f"at {istep} KMC step!")
                logstr += "\n"
                LogWriter.write_data(logstr)
        else:
            DataSPs = thisRestart.DataSPs
            AVitags = thisRestart.AVitags
            df_delete_SPs = thisRestart.df_delete_SPs
            undo_idavs = thisRestart.undo_idavs
            finished_AVs = thisRestart.finished_AVs
            thisRestart = None
            if rank_world == 0:
                thisdf = pd.DataFrame()
                for i in range(len(DataSPs.df_SPs)):
                    if len(DataSPs.df_SPs[i]) > 0:
                        thisdf = pd.concat([thisdf, DataSPs.df_SPs[i]], ignore_index=True)
                if len(thisdf) == 0:
                    thisdf = pd.DataFrame(columns=SP_DATA_HEADER)
                DFWriter.write_SPs(thisdf, idstart=0, mode="w")
                DFWriter.write_deleted_SPs(df_delete_SPs, idstart=0, mode="w")
                logstr = "There are " + str(len(undo_idavs)) + " undo defects (active volumes) in data!"
                LogWriter.write_data(logstr)

        comm_world.Barrier()

        DataSPs, AVitags, df_delete_SPs = dataSPS.data_find_saddlepoints(istep, thissett, seakmcdata, DefectBank_list,
                                                                         thisSuperBasin, Eground,
                                                                         DataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                         finished_AVs, simulation_time, object_dict)

        seakmcdata.to_atom_style()
        seakmcdata.velocities = None
        seakmcdata.defects = None
        seakmcdata.def_atoms = []
        seakmcdata.atoms_ghost = None
        seakmcdata.natoms_ghost = 0

        os.chdir(THIS_PATH)

        if thissett.saddle_point["CalBarrsInData"]:
            DataSPs, df_delete_this = myRecal.recalibrate_energy(thissett, DataSPs, seakmcdata, AVitags, Eground,
                                                                 object_dict)
            if rank_world == 0:
                df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)

        df_delete_this = None
        if DataSPs.nSP <= 0:
            errormsg = "###  No saddle point found! ###"
            error_exit(errormsg)

        if rank_world == 0:
            if (thissett.system["Restart"]["WriteRestart"] and
                    istep % thissett.system["Restart"]["KMCStep4Restart"] == 0):
                thisRestart = RESTART(istep, seakmcdata.ndefects, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                                      DataSPs, AVitags, df_delete_SPs, np.array([], dtype=int), simulation_time)
                thisRestart.to_file()
                thisRestart = None

            logstr = "\n" + "In KMC step ..."
            LogWriter.write_data(logstr)
            thisExports["ground_energy"] = Eground
            simulation_time, thiskmc, thisSuperBasin, thisExports = dataKMC.run_KMC(istep, thisSuperBasin,
                                                                                    seakmcdata, AVitags, DataSPs,
                                                                                    thissett, simulation_time,
                                                                                    thisExports, LogWriter)
            thisSummary.update_data(thisExports)
            thisSummary.write_data()

            this_simulation_time = thiskmc.timeelapse
            dataout.write_prob_to_file(thissett.visual, thiskmc, DataSPs, istep, out_paths[4],
                                       VerySmallNumber=thissett.system["VerySmallNumber"])
            sel_SPs = dataout.get_sel_SPs_for_out(thissett.visual, thiskmc, DataSPs)
        else:
            sel_SPs = None
            this_simulation_time = None
            thisExports = None

        comm_world.Barrier()
        sel_SPs = comm_world.bcast(sel_SPs, root=0)
        this_simulation_time = comm_world.bcast(this_simulation_time, root=0)
        thisExports = comm_world.bcast(thisExports, root=0)
        if len(sel_SPs) > 0:
            dataout.visualize_data_SPs(thissett.visual, seakmcdata, AVitags, DataSPs, sel_SPs, istep, out_paths[1])
        else:
            if rank_world == 0: dataout.visualize_data_SPs_Superbasin(thissett.visual, thiskmc, thisSuperBasin, istep,
                                                                      out_paths[1])

        comm_world.Barrier()
        DataSPs = None
        sel_SPs = None

        if rank_world == 0:
            last_de_center = thiskmc.update_last_defect_center(thisSuperBasin)
            if isinstance(thissett.active_volume["DefectCenter4RT_SetMolID"], list):
                last_de_center = thissett.active_volume["DefectCenter4RT_SetMolID"]
            seakmcdata = thiskmc.update_coords4relaxation(thisSuperBasin)
            thisSuperBasin.prepare_next(thissett.kinetic_MC)
            AVitags = None
            thiskmc = None
            logstr = "Relaxing the structure ..."
        else:
            last_de_center = None
            AVitags = None
            thisSuperBasin = None
            AVitags = None
            thiskmc = None
            seakmcdata = None

        comm_world.Barrier()
        seakmcdata = comm_world.bcast(seakmcdata, root=0)
        last_de_center = comm_world.bcast(last_de_center, root=0)

        nproc_task = thissett.force_evaluator['nproc']
        [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("OPT", force_evaluator, seakmcdata, 1,
                                                                               nactive=seakmcdata.natoms,
                                                                               nproc_task=nproc_task,
                                                                               thisExports=thisExports)

        if rank_world == 0:
            for f in os.listdir("Runner_0/"):
                for i in range(len(thissett.force_evaluator["OutFileHeaders"])):
                    if thissett.force_evaluator["OutFileHeaders"][i] in f:
                        shutil.copy("Runner_0/" + f, out_paths[1] + "/KMC_" + str(istep + 1) + "_" + f)

        seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
        seakmcdata.assert_settings(thissett)
        seakmcdata.to_atom_style()
        seakmcdata.velocities = None
        if rank_world == 0:
            if thissett.visual["Write_Data_SPs"]["Write_KMC_Data"]:  seakmcdata.to_lammps_data(
                out_paths[1] + "/" + "KMC_" + str(istep + 1) + ".dat", to_atom_style=True)

            tockmc = time.time()
            logstr += "\n" + "KMC " + str(istep) + "th step is finished."
            logstr += ("\n" +
                       f"Time step for {istep} KMC step: "
                       f"{round(this_simulation_time, thissett.system['float_precision'])} ps")
            logstr += ("\n" +
                       f"Summed time steps after {istep} KMC step: "
                       f"{round(simulation_time, thissett.system['float_precision'])} ps")
            logstr += ("\n" +
                       f"Real time cost for {istep} KMC step: "
                       f"{round(tockmc - tickmc, thissett.system['float_precision'])} s")
            logstr += "\n" + "==================================================================="
            LogWriter.write_data(logstr)

        comm_world.Barrier()

    return simulation_time
