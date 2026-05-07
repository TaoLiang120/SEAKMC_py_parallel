import numpy as np

from mpi4py import MPI

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def get_2D_task_distribution(inrow, incol, ntask_time):
    ntot = inrow * incol
    processing_tasks = np.array([], dtype=int)

    idstart = 0
    while idstart < ntot:
        irowstart = int(idstart / incol)
        irowend = min(irowstart + ntask_time, inrow)
        ids = np.arange(idstart, (irowend - irowstart) * incol + idstart)
        ids = ids.reshape([irowend - irowstart, incol])
        processing_tasks = np.append(processing_tasks, ids.T.flatten())
        idstart += (irowend - irowstart) * incol
    return processing_tasks


def get_proc_partition(ntot, size, nmin_rank=1):
    n_rank = int(ntot / size)
    n_rank = max(n_rank, nmin_rank)
    rank_last = int(ntot / n_rank)
    if rank_last == 0:
        n_rank_last = ntot
    elif rank_last < size:
        n_rank_last = ntot - rank_last * n_rank
    else:
        n_rank_last = ntot - rank_last * n_rank + n_rank
        rank_last = size - 1

    return n_rank, rank_last, n_rank_last


def get_ntask_time(nproc_task, start_proc=0, thiscomm=None):
    if thiscomm is not None:
        pass
    else:
        thiscomm = MPI.COMM_WORLD
    size_local = thiscomm.Get_size()
    #rank_local = thiscomm.Get_rank()
    if size_local < nproc_task + start_proc:
        print("The number of cores must be greater than the number of communicators.")
        MPI.COMM_WORLD.Abort()
    ntask_time = int((size_local - start_proc) / nproc_task)

    return ntask_time


def split_communicator(nproc_task, start_proc=0, thiscomm=None):
    if thiscomm is not None:
        pass
    else:
        thiscomm = MPI.COMM_WORLD
    size_local = thiscomm.Get_size()
    rank_local = thiscomm.Get_rank()
    if rank_local < start_proc:
        thiscolor = size_local
    else:
        thiscolor = int((rank_local - start_proc) / nproc_task)
    #thiskey = (rank_local - start_proc) % nproc_task
    comm_split = thiscomm.Split(thiscolor)
    return comm_split, thiscolor

def get_COMM_info(nproc_task, start_proc=0):
    comm_world = MPI.COMM_WORLD
    size_world = comm_world.Get_size()
    if nproc_task == size_world:
        COMM_dict = {"isSplit": False, "color": 0, "thiscomm": comm_world}
    else:
        comm_split, thiscolor = split_communicator(nproc_task, start_proc=start_proc, thiscomm=None)
        COMM_dict = {"isSplit": True, "color": thiscolor,  "thiscomm": comm_split}
    return COMM_dict