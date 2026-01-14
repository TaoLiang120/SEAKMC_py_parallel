import sys
from mpi4py import MPI
rank_world = MPI.COMM_WORLD.Get_rank()
def error_exit(error_str):
    MPI.Finalize()
    if rank_world == 0:
        print("=== Error Message===")
        print(error_str)
        print("=== End of Error ===")
    sys.exit(1)
