import os

from mpi4py import MPI

import seakmc_p.general.General as mygen
import seakmc_p.process.DataDyn as mydatadyn
from seakmc_p.core.data import SeakmcData
from seakmc_p.restart.Restart import RESTART
import seakmc_p.mpiconf.MPIconf as mympi
from seakmc_p.mpiconf.error_exit import error_exit

def load_RESTART(Restartsett):
    thisRestart = None
    if Restartsett["LoadRestart"]:
        filename = Restartsett["LoadFile"]
        if isinstance(filename, str):
            thisRestart = RESTART.from_file(filename)
        else:
            FileHeader = "RESTART_istep_"
            fapp = ".restart"
            files = []
            for file in os.listdir(os.getcwd()):
                if FileHeader in file and fapp in file:
                    thisstr = file.replace(FileHeader, "")
                    thisstr = thisstr.replace(fapp, "")
                    thisstrs = thisstr.split("_")
                    try:
                        istep_this = int(thisstrs[0])
                        finished_AVs = int(thisstrs[1])
                        files.append((istep_this, finished_AVs))
                    except:
                        pass
            if len(files) > 0:
                fsorted = sorted(files, key=lambda t: (t[0], t[1]), reverse=True)
                filename = FileHeader + str(fsorted[0][0]) + "_" + str(fsorted[0][1]) + fapp
                thisRestart = RESTART.from_file(filename)
            else:
                thisRestart = None
    return thisRestart


def initial_data_dynamics(thissett, seakmcdata, force_evaluator, LogWriter, **COMM_args):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()
    ntask_tot = 1
    nproc_task = thissett.force_evaluator['nproc']
    if thissett.data["MoleDyn"]:
        if rank_world == 0:
            logstr = "Molecular dynamics simulation of the initial structure ..."
            LogWriter.write_data(logstr)

        comm_world.Barrier()
        [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("DATAMD", force_evaluator, seakmcdata,
                                                                               ntask_tot,
                                                                               nactive=seakmcdata.natoms,
                                                                               nproc_task=nproc_task, thisExports=None,
                                                                               **COMM_args)

        if rank_world == 0:
            seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
            seakmcdata.assert_settings(thissett)
            seakmcdata.to_atom_style()
            seakmcdata.velocities = None
        else:
            seakmcdata = None

        comm_world.Barrier()
        seakmcdata = comm_world.bcast(seakmcdata, root=0)

        [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("DATAOPT", force_evaluator, seakmcdata,
                                                                               ntask_tot,
                                                                               nactive=seakmcdata.natoms,
                                                                               nproc_task=nproc_task, thisExports=None,
                                                                               **COMM_args)

        if rank_world == 0:
            seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
            seakmcdata.assert_settings(thissett)
            seakmcdata.to_atom_style()
            seakmcdata.velocities = None
        else:
            seakmcdata = None

        comm_world.Barrier()
        seakmcdata = comm_world.bcast(seakmcdata, root=0)

    elif not thissett.data["Relaxed"]:
        if rank_world == 0:
            logstr = "Relaxing the initial structure ..."
            LogWriter.write_data(logstr)

        comm_world.Barrier()
        [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("DATAOPT", force_evaluator, seakmcdata,
                                                                               ntask_tot,
                                                                               nactive=seakmcdata.natoms,
                                                                               nproc_task=nproc_task, thisExports=None,
                                                                               **COMM_args)

        if rank_world == 0:
            if not isValid:
                LogWriter.write_data(errormsg)
                error_exit(errormsg)
            seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
            seakmcdata.assert_settings(thissett)
            seakmcdata.to_atom_style()
            seakmcdata.velocities = None
        else:
            seakmcdata = None

        comm_world.Barrier()
        seakmcdata = comm_world.bcast(seakmcdata, root=0)
    else:
        [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("DATAMD0", force_evaluator, seakmcdata,
                                                                               ntask_tot,
                                                                               nactive=seakmcdata.natoms,
                                                                               nproc_task=nproc_task, thisExports=None,
                                                                               **COMM_args)
    return seakmcdata, Eground


def preprocess(thissett):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()
    Eground = 0.0
    thisRestart = load_RESTART(thissett.system["Restart"])
    if rank_world == 0:
        nproc_task_min = max(1, thissett.spsearch["force_evaluator"]["nproc"])
        ntask_time_max = int(size_world / nproc_task_min)
        for i in range(ntask_time_max):
            foldn = "Runner_" + str(i)
            os.makedirs(foldn, exist_ok=True)
        object_dict = mygen.object_maker(thissett, thisRestart)
    else:
        object_dict = None

    comm_world.Barrier()
    object_dict = comm_world.bcast(object_dict, root=0)

    out_paths = object_dict['out_paths']
    LogWriter = object_dict['LogWriter']
    force_evaluator = object_dict['force_evaluator']

    nproc_task = thissett.force_evaluator['nproc']
    COMM_args = mympi.get_COMM_info(nproc_task, start_proc=0)
    GPU_args = thissett.force_evaluator["GPU"]

    force_evaluator.init_binary(comm=COMM_args["thiscomm"],
                     Screen=thissett.force_evaluator['Screen'], Log=thissett.force_evaluator['LogFile'], **GPU_args)

    if thisRestart is None:
        seakmcdata = SeakmcData.from_file(thissett.data['FileName'], atom_style=thissett.data['atom_style'])
        seakmcdata.assert_settings(thissett)
        seakmcdata.to_atom_style()
        seakmcdata.velocities = None

        if rank_world == 0:
            logstr = "Successfully loading input and structure ..."
            LogWriter.write_data(logstr)
        seakmcdata, Eground = initial_data_dynamics(thissett, seakmcdata, force_evaluator, LogWriter, **COMM_args)
        if thissett.visual["Write_Data_SPs"]["Write_KMC_Data"]:
            if rank_world == 0:
                seakmcdata.to_lammps_data(out_paths[1] + "/" + "KMC_" + str(0) + ".dat", to_atom_style=True)
    else:
        seakmcdata = thisRestart.seakmcdata
        seakmcdata.assert_settings(thissett)
        Eground = thisRestart.Eground
        istep_this = thisRestart.istep_this
        simulation_time = thisRestart.simulation_time
        if Eground is None or Eground == 0.0:
            [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("DATAMD0", force_evaluator,
                                                                                   seakmcdata, 1,
                                                                                   nactive=seakmcdata.natoms,
                                                                                   nproc_task=thissett.force_evaluator[
                                                                                       'nproc'], thisExports=None, **COMM_args)

        if thissett.visual["Write_Data_SPs"]["Write_KMC_Data"]:
            if rank_world == 0:
                seakmcdata.to_lammps_data(out_paths[1] + "/" + "KMC_" + str(istep_this) + ".dat", to_atom_style=True)

    comm_world.Barrier()
    force_evaluator.close()
    if COMM_args["isSplit"]:
        COMM_args["thiscomm"].Free()
    comm_world.Barrier()

    return seakmcdata, object_dict, Eground, thisRestart
