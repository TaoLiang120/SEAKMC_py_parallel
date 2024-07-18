import os
import pickle

from mpi4py import MPI

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


class RESTART:
    def __init__(
            self,
            istep_this,
            finished_AVs,
            DefectBank_list,
            thisSuperBasin,
            seakmcdata,
            Eground,
            DataSPs,
            AVitags,
            df_delete_SPs,
            undo_idavs,
            simulation_time,
    ):
        self.istep_this = istep_this
        self.finished_AVs = finished_AVs
        self.DefectBank_list = DefectBank_list
        self.thisSuperBasin = thisSuperBasin
        self.seakmcdata = seakmcdata
        self.Eground = Eground
        self.DataSPs = DataSPs
        self.AVitags = AVitags
        self.df_delete_SPs = df_delete_SPs
        self.undo_idavs = undo_idavs
        self.simulation_time = simulation_time

    def __str__(self):
        return f"Restart istep: {self.istep_this} and finished AVs: {self.finished_AVs}."

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_file(cls, filename):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                try:
                    thisRestart = pickle.load(f)
                except:
                    print("Cannot load restart file!")
                    comm_world.Abort(rank_world)
        else:
            print(filename + " is not existed.")
            comm_world.Abort(rank_world)
        return thisRestart

    def to_file(self):
        filename = "RESTART_istep_" + str(self.istep_this) + "_" + str(self.finished_AVs) + ".restart"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
