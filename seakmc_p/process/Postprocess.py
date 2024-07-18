import os
import shutil
import time

from mpi4py import MPI

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def postprocess(tic, thissett, object_dict, simulation_time):
    comm_world.Barrier()
    LogWriter = object_dict['LogWriter']
    if rank_world == 0:
        folds = os.listdir()
        for fold in folds:
            if "Runner_" in fold: shutil.rmtree(fold)
        for f in thissett.system["TempFiles"]:
            if os.path.isfile(f): os.remove(f)

    if rank_world == 0:
        toc = time.time()
        logstr = "\n" + (f"Total KMC time steps for this simulation: "
                         f"{round(simulation_time, thissett.system['float_precision'])} ps")
        logstr += "\n" + "Real time cost for this simulation:" + str(
            round(toc - tic, thissett.system['float_precision'])) + " s"
        logstr += "\n" + "==================================================================="
        LogWriter.write_data(logstr)
