from mpi4py import MPI

import seakmc_p.mpiconf.MPIconf as mympi

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


def data_dynamics(purpose, force_evaluator, data, ntask_tot, nactive=None, nproc_task=1, thisExports=None):
    if nactive is None:
        try:
            nactive = data.nactive
        except:
            nactive = data.natoms

    start_proc = 0
    ntask_time = mympi.get_ntask_time(nproc_task, start_proc=start_proc, thiscomm=None)
    comm_split, thiscolor = mympi.split_communicator(nproc_task, start_proc=start_proc, thiscomm=None)

    ntask_left = ntask_tot
    ntask_time = min(ntask_time, ntask_left)
    itask_start = 0
    while ntask_left > 0:
        if thiscolor < ntask_time:
            [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner(purpose, data, thiscolor,
                                                                                      nactive=nactive,
                                                                                      thisExports=thisExports,
                                                                                      comm=comm_split)
        else:
            Eground = None
            relaxed_coords = None
            isValid = None
            errormsg = None

        comm_world.Barrier()

        itask_start += ntask_time
        ntask_left = ntask_left - ntask_time
        ntask_time = min(ntask_time, ntask_left)

    comm_split.Free()
    comm_world.Barrier()
    Eground = comm_world.bcast(Eground, root=0)
    relaxed_coords = comm_world.bcast(relaxed_coords, root=0)
    isValid = comm_world.bcast(isValid, root=0)
    errormsg = comm_world.bcast(errormsg, root=0)
    if not isValid and rank_world == 0:
        print(errormsg)
        comm_world.Abort(rank_world)

    return [Eground, relaxed_coords, isValid, errormsg]
