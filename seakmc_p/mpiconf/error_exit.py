import sys
from mpi4py import MPI
def error_exit(error_str):
    print("=== Error Message ===")
    print(error_str)
    print("=== End of Error ===")
    sys.stdout.flush()
    MPI.COMM_WORLD.Abort(1)
    #MPI.Finalize()
    #sys.exit(0)
