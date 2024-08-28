import ctypes
import os

import numpy as np
from lammps import lammps
from monty.io import zopen
from mpi4py import MPI

from seakmc_p.input.Input import export_Keys

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()


class PyLammpsRunner(object):
    def __init__(self, sett):
        self.name = 'pylammps'
        self.sett = sett
        self.dynmat = "dynmat.dat"
        self.input_file = "in.lammps"

        if isinstance(self.sett.potential['Path2Pot'], str):
            self.path_to_pot = self.sett.potential['Path2Pot']
        else:
            self.path_to_pot = os.getcwd()

        if isinstance(self.sett.potential["FileName"], str):
            if not os.path.isfile(os.path.join(self.path_to_pot, self.sett.potential["FileName"])):
                print(f"Cannot find {os.path.join(self.path_to_pot, self.sett.potential['FileName'])} !")
                comm_world.Abort(rank_world)

    def run_runner(self, purpose, data, thiscolor, nactive=None,
                   thisExports=[], comm=None):

        purpose = purpose.upper()
        nproc_task = self.get_nproc_task(purpose)
        self.this_path = os.getcwd()
        self.relative_path = "Runner_" + str(thiscolor)

        if comm is None: comm = MPI.COMM_WORLD
        rank_local = comm.Get_rank()
        if rank_local == 0:
            data.to_lammps_data(self.relative_path + "/tmp0.dat", to_atom_style=True, distance=12)
            rinputs = self.preparation(purpose, data, thiscolor, nactive=nactive,
                                       thisExports=thisExports, nproc=nproc_task, comm=comm)
        else:
            rinputs = None

        comm.Barrier()

        errormsg = ""
        isValid = True
        total_energy = 0.0
        relaxed_coords = []
        rinputs = comm.bcast(rinputs, root=0)
        self.init_binary(nproc=nproc_task, comm=comm,
                         Screen=self.sett.force_evaluator['Screen'], Log=self.sett.force_evaluator['LogFile'])

        total_energy, relaxed_coords = self.execute_runner(rinputs)

        if rank_local == 0:
            if total_energy is None or relaxed_coords is None:
                isValid = False
                errormsg = (f"Error on running PYLAMMPS - "
                            f"purpose: {purpose} datatype:{type(data)} thiscolor: {thiscolor} nactive:{nactive}!")
        else:
            isValid = None
            errormsg = None

        comm.Barrier()

        isValid = comm.bcast(isValid, root=0)
        errormsg = comm.bcast(errormsg, root=0)
        self.close()

        return [total_energy, relaxed_coords, isValid, errormsg]

    def init_spsearch_runner(self, data, thiscolor, nactive, comm=None):
        purpose = "SPS"
        nproc_task = self.get_nproc_task(purpose)
        self.this_path = os.getcwd()
        self.relative_path = "Runner_" + str(thiscolor)

        if comm is None: comm = MPI.COMM_WORLD
        rank_local = comm.Get_rank()
        if rank_local == 0:
            data.to_lammps_data(self.relative_path + "/tmp0.dat", to_atom_style=True, distance=12)
            rinputs = self.preparation(purpose, data, thiscolor, nactive=nactive, nproc=nproc_task, comm=comm)
        else:
            rinputs = None

        comm.Barrier()
        rinputs = comm.bcast(rinputs, root=0)
        errormsg = ""
        isValid = True
        total_energy = 0.0
        relaxed_coords = []
        self.init_binary(nproc=nproc_task, comm=comm,
                         Screen=self.sett.force_evaluator['Screen'], Log=self.sett.force_evaluator['LogFile'])

        total_energy, relaxed_coords = self.execute_runner(rinputs)

        if rank_local == 0:
            if total_energy is None or relaxed_coords is None:
                isValid = False
                errormsg = f"Error on initializing PYLAMMPS!"
                errormsg += ("\n" +
                             f"Job - purpose:{purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")
        else:
            isValid = None
            errormsg = None

        comm.Barrier()

        isValid = comm.bcast(isValid, root=0)
        errormsg = comm.bcast(errormsg, root=0)
        return [total_energy, [], isValid, errormsg]

    def get_spsearch_forces(self, coords, data, thiscolor, nactive, comm=None):
        errormsg = ""
        isValid = True
        total_energy = 0.0
        forces = []
        purpose = "SPS"
        x = coords.flatten()
        c_float_p = ctypes.POINTER(ctypes.c_double)
        x_p = x.ctypes.data_as(c_float_p)
        try:
            self.bin.scatter_atoms("x", 1, 3, x_p)
            self.bin.command("run   0")
            #encalc = self.bin.extract_compute("av_pe", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
            encalc = self.bin.get_thermo("etotal")
            forces = self.bin.gather_atoms("f", 1, 3)
            forces = np.ctypeslib.as_array(forces)
        except:
            isValid = False
            errormsg = f"Error on getting forces from PYLAMMPS!"
            errormsg += ("\n" +
                         f"Job - purpose:{purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")
        return [encalc, forces, isValid, errormsg]

    def init_binary(self, nproc=1, comm=None, Screen=False, Log=False):
        args = []
        if not Screen: args = ["-screen", "none"]
        if isinstance(Log, str):
            args += ["-log", Log]
        else:
            args += ["-log", "none"]
        self.bin = lammps(cmdargs=args, comm=comm)

    def execute_runner(self, rinputs):
        try:
            self.bin.file(self.relative_path + "/" + self.input_file)
            self.bin.command("run 0")
            etotal = self.bin.get_thermo("etotal")
            coords = self.bin.gather_atoms("x", 1, 3)
            coords = np.ctypeslib.as_array(coords)
        except:
            etotal = None
            coords = None
        return etotal, coords

    def preparation(self, purpose, data, thiscolor, nactive=None, thisExports=[], nproc=1, comm=None):
        purpose = purpose.upper()
        if nactive is None:
            try:
                nactive = data.nactive
            except:
                nactive = data.natoms
        if purpose == "DATAMD":
            if isinstance(self.sett.data["RinputMD"], str):
                rinputs = self.modify_input_script(self.sett.data["RinputMD"], thiscolor, nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "DATAMD0":
            if isinstance(self.sett.data["RinputMD0"], str):
                rinputs = self.modify_input_script(self.sett.data["RinputMD0"], thiscolor, nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "DATAOPT" or purpose == "DATARELAX":
            if isinstance(self.sett.data["RinputOpt"], str):
                rinputs = self.modify_input_script(self.sett.data["RinputOpt"], thiscolor, nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "MD0" or "RECAL" in purpose:
            if isinstance(self.sett.force_evaluator["RinputMD0"], str):
                rinputs = self.modify_input_script(self.sett.force_evaluator["RinputMD0"], thiscolor, nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "OPT" or purpose == "RELAX":
            if isinstance(self.sett.force_evaluator["RinputOpt"], str):
                rinputs = self.modify_input_script(self.sett.force_evaluator["RinputOpt"], thiscolor, nactive)
                rinputs = self.ImportValue4RinputOpt(rinputs, thisExports)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "SPSDYNMAT" or purpose == "DYNMAT":
            if isinstance(self.sett.spsearch["force_evaluator"]["RinputDM"], str):
                rinputs = self.modify_input_script(self.sett.spsearch["force_evaluator"]["RinputDM"], thiscolor,
                                                   nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "SPS":
            if isinstance(self.sett.spsearch["force_evaluator"]["Rinput"], str):
                rinputs = self.modify_input_script(self.sett.spsearch["force_evaluator"]["Rinput"], thiscolor, nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)
        elif purpose == "SPSOPT" or purpose == "SPSRELAX":
            if isinstance(self.sett.spsearch["force_evaluator"]["RinputOpt"], str):
                rinputs = self.modify_input_script(self.sett.spsearch["force_evaluator"]["RinputOpt"], thiscolor,
                                                   nactive)
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive)

        if os.path.isfile(self.relative_path + "/" + self.input_file):
            os.remove(self.relative_path + "/" + self.input_file)
        with zopen(self.relative_path + "/" + self.input_file, "wt") as f:
            f.write("\n".join(rinputs) + "\n")

        return rinputs

    def get_default_inputs(self, purpose, data, thiscolor, nactive):
        def get_fixtypes_lines(lines, FixTypes_dict):
            for fixtype in FixTypes_dict:
                thisFixAxes = FixTypes_dict[fixtype]
                lines.append("group gfixt%d type %d" % (fixtype, fixtype))
                lines.append("group gfixt%d intersect gfixt%d gactive" % (fixtype, fixtype))
                lines.append("group gactive subtract gactive gfixt%d" % fixtype)
                forcestr = ""
                for i in range(3):
                    if i in thisFixAxes:
                        forcestr += " 0.0"
                    else:
                        forcestr += " NULL"
                lines.append("fix ffix%d gfixt%d setforce %s" % (fixtype, fixtype, forcestr))
            return lines

        lines = []
        lines.append("units         %s" % (self.sett.data["units"]))
        lines.append("dimension     %d" % (self.sett.data["dimension"]))
        lines.append("atom_style    %s" % (self.sett.data["atom_style_after"]))
        lines.append("atom_modify   %s" % "map array")

        if "SPS" in purpose or "DYNMAT" in purpose:
            lines.append("boundary      %s" % (self.sett.active_volume["boundary"]))
            if isinstance(self.sett.spsearch["force_evaluator"]["partition"], str):
                lines.append("partition      %s" % (self.sett.spsearch["force_evaluator"]["partition"]))
            if isinstance(self.sett.spsearch["force_evaluator"]["processors"], str):
                lines.append("processors     %s" % (self.sett.spsearch["force_evaluator"]["processors"]))
        else:
            lines.append("boundary      %s" % (self.sett.data["boundary"]))
            if isinstance(self.sett.force_evaluator["partition"], str):
                lines.append("partition      %s" % (self.sett.force_evaluator["partition"]))
            if isinstance(self.sett.force_evaluator["processors"], str):
                lines.append("processors     %s" % (self.sett.force_evaluator["processors"]))

        if self.sett.potential["OpenKIM"]["OpenKIM"]:
            lines.append("kim init    %s" % (self.sett.potential["OpenKIM"]["kim_init"]))
        else:
            lines.append("pair_style    %s" % (self.sett.potential["pair_style"]))
        lines.append("read_data     Runner_" + str(thiscolor) + "/tmp0.dat")

        se = ""
        for i in range(len(self.sett.potential["species"])):
            se += self.sett.potential["species"][i] + " "
        if self.sett.potential["OpenKIM"]["OpenKIM"]:
            lines.append("kim interactions    %s" % (self.sett.potential["OpenKIM"]["kim_interaction"]))
            if isinstance(self.sett.potential["OpenKIM"]["kim_param"], str):
                lines.append("kim param    %s" % (self.sett.potential["OpenKIM"]["kim_param"]))
        else:
            if isinstance(self.sett.potential["pair_coeff"], str):
                lines.append("pair_coeff    %s" % (self.sett.potential["pair_coeff"]))
            elif isinstance(self.sett.potential["pair_coeff"], list):
                for line in self.sett.potential["pair_coeff"]:
                    lines.append("pair_coeff    %s" % (line))
            elif isinstance(self.sett.potential["FileName"], str):
                f = os.path.join(self.path_to_pot, self.sett.potential["FileName"])
                line = "pair_coeff    *  *  " + f + "  " + se
                lines.append(line)

        lines.append("neighbor 2.0 bin")
        lines.append("neigh_modify every 1 delay 0 check yes")
        lines.append("timestep    %f" % (self.sett.force_evaluator["timestep"]))
        lines.append("compute     pe  all pe")
        lines.append("compute     ke  all ke")

        if purpose == "DATAMD":
            lines.append("fix 1 all  nvt   temp  300.0  300.0   100.0")
            lines.append("run     %d" % (self.sett.force_evaluator["NSteps4Relax"]))
            #lines.append("print   etotal=$(pe+ke)")
            lines.append("write_data     Runner_" + str(thiscolor) + "/tmp1.dat")
        elif "MD0" in purpose or "RECAL" in purpose:
            lines.append("fix 1 all  nvt   temp  300.0  300.0   100.0")
            lines.append("run     0")
            #lines.append("print   etotal=$(pe+ke)")
            #lines.append("write_data     Runner_" + str(thiscolor)+"/tmp1.dat")
        elif purpose == "OPT" or purpose == "RELAX" or purpose == "DATAOPT" or purpose == "DATARELAX":
            Steps = self.sett.force_evaluator["NSteps4Relax"]
            if purpose == "OPT" or purpose == "RELAX":
                if self.sett.force_evaluator["Relaxation"]["BoxRelax"]:
                    lines.append("fix fbox all box/relax iso 1.0 vmax 0.001")
            else:
                if self.sett.data["BoxRelax"]:
                    lines.append("fix fbox all box/relax iso 1.0 vmax 0.001")
            lines.append("minimize  0.0 1.0e-6 %d   %d" % (max(100, int(Steps / 10)), Steps))
            if (self.sett.force_evaluator["Relaxation"]["InitTemp4Opt"] >
                    self.sett.force_evaluator["Relaxation"]["TargetTemp4NVT"]):
                lines.append("velocity     all create %f 10 dist uniform" % (
                    self.sett.force_evaluator["Relaxation"]["InitTemp4Opt"]))
                t = self.sett.force_evaluator["Relaxation"]["TargetTemp4NVT"]
                lines.append("fix 1nvt all nvt   temp  %f  %f   %f " % (
                    t, t, 0.33 * self.sett.force_evaluator["Relaxation"]["InitTemp4Opt"]))
                lines.append("run %d" % (self.sett.force_evaluator["Relaxation"]["NVTSteps4Opt"]))
                lines.append("unfix 1nvt")
                lines.append("minimize     0.0 1.0e-6 %d  %d" % (max(100, int(Steps / 10)), Steps))
            lines.append("run    0")
            #lines.append("print   etotal=$(pe+ke)")
            lines.append("write_data     Runner_" + str(thiscolor) + "/tmp1.dat")

        elif "DYNMAT" in purpose:
            lines.append("group        gactive id <= %d" % nactive)
            if self.sett.spsearch["FixTypes_dict"] is not None:
                lines = get_fixtypes_lines(lines, self.sett.spsearch["FixTypes_dict"])
            lines.append("dynamical_matrix gactive regular %f file %s " % (
                self.sett.dynamic_matrix["displacement"], "Runner_" + str(thiscolor) + "/" + self.dynmat))
            lines.append("run   0")
            #lines.append("print   etotal=$(pe+ke)")

        elif purpose == "SPS":
            lines.append("group        gactive id <= %d" % nactive)
            lines.append("group        gfixed subtract all gactive")
            if self.sett.spsearch["FixTypes_dict"] is not None:
                lines = get_fixtypes_lines(lines, self.sett.spsearch["FixTypes_dict"])
            lines.append("fix          freeze gfixed setforce 0.0 0.0 0.0")
            lines.append("run    0")
            #lines.append("print   etotal=$(pe+ke)")

        elif purpose == "SPSOPT" or purpose == "SPSRELAX":
            Steps = self.sett.force_evaluator["NSteps4Relax"]
            lines.append("group        gactive id <= %d" % nactive)
            lines.append("group        gfixed subtract all gactive")
            if self.sett.spsearch["FixTypes_dict"] is not None:
                lines = get_fixtypes_lines(lines, self.sett.spsearch["FixTypes_dict"])
            lines.append("fix          freeze gfixed setforce 0.0 0.0 0.0")
            lines.append("minimize     0.0 1.0e-6 %d   %d" % (max(100, int(Steps / 10)), Steps))
            if self.sett.spsearch["LocalRelax"]["InitTemp4Opt"] > self.sett.spsearch["LocalRelax"]["TargetTemp4NVT"]:
                t0 = self.sett.spsearch["LocalRelax"]["InitTemp4Opt"]
                t = self.sett.spsearch["LocalRelax"]["TargetTemp4NVT"]
                lines.append("velocity     gactive create %f 10 dist uniform" % t0)
                lines.append("fix 1nvt all nvt   temp  %f  %f   %f " % (t, t, 0.33 * t0))
                lines.append("run %d" % (self.sett.spsearch["LocalRelax"]["NVTSteps4Opt"]))
                lines.append("unfix 1nvt")
                lines.append("minimize     0.0 1.0e-6 %d  %d" % (max(100, int(Steps / 10)), Steps))
            #lines.append("run    0")
            #lines.append("print   etotal=$(pe+ke)")
        return lines

    def get_nproc_task(self, purpose):
        if "SPS" in purpose or "DYNMAT" in purpose:
            nproc_task = self.sett.spsearch['force_evaluator']['nproc']
        elif purpose == "MD0" or "RECAL" in purpose:
            nproc_task = self.sett.force_evaluator['nproc4ReCal']
        else:
            nproc_task = self.sett.force_evaluator['nproc']
        return nproc_task

    def modify_input_script(self, input_file, thiscolor, nactive):
        with open(input_file, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            if "read_data " == lines[i][0:10]:
                lines[i] = "read_data    " + "Runner_" + str(thiscolor) + "/tmp0.dat"
            if "dump " == lines[i][0:5]:
                thislines = list(filter(None, list(map(lambda strings: strings.strip(), lines[i].split(" ")))))
                if "/" in thislines[5]:
                    str5 = thislines[5].split("/")
                    thislines[5] = "Runner_" + str(thiscolor) + "/" + str5[-1]
                else:
                    thislines[5] = "Runner_" + str(thiscolor) + "/" + thislines[5]
                line = thislines[0]
                for j in range(1, len(thislines)):
                    line += " " + thislines[j]
                lines[i] = line
            if "dynamical_matrix " == lines[i][0:17]:
                lines[i] = "dynamical_matrix gactive regular %f file %s " % (
                    self.sett.dynamic_matrix["displacement"], "Runner_" + str(thiscolor) + "/" + self.dynmat)
            if "write_data " == lines[i][0:11]:
                lines[i] = "write_data    " + "Runner_" + str(thiscolor) + "/tmp1.dat"
            if "group " == lines[i][0:6] and " id <= " in lines[i]:
                thislines = lines[i].split("id <= ")
                lattlines = list(filter(None, list(map(lambda strings: strings.strip(), thislines[1].split(" ")))))
                lattlines[0] = str(nactive)
                line = thislines[0] + " id <="
                for j in range(0, len(lattlines)):
                    line += " " + lattlines[j]
                lines[i] = line
        return lines

    def ImportValue4RinputOpt(self, rinputs, thisExports=[]):
        isValid = True
        if not self.sett.force_evaluator['ImportValue4RinputOpt']: isValid = False
        if thisExports is None:
            isValid = False
        elif len(thisExports) == 0:
            isValid = False
        if isValid:
            KEYS = self.sett.force_evaluator['Keys4ImportValue4RinputOpt']
            InKeys = []
            InVals = []
            n = len(KEYS)
            for i in range(n):
                thiskeys = KEYS[i]
                m = len(thiskeys)
                if m == 2:
                    if thiskeys[1] in export_Keys:
                        InKeys.append(thiskeys[0])
                        InVals.append(thisExports[thiskeys[1]])

            n = len(InKeys)
            for i in range(len(rinputs)):
                line = rinputs[i]
                for ik in range(n):
                    key = InKeys[ik]
                    istart = line.find(key)
                    if istart != -1:
                        thislines = line.split(key)
                        afters = list(filter(None, list(map(lambda strings: strings.strip(), thislines[1].split(" ")))))
                        na = len(afters)
                        if istart == 0:
                            newline = key + " " + str(InVals[ik])
                        else:
                            newline = thislines[0] + " " + key + " " + str(InVals[ik])
                        if na > 1:
                            for j in range(1, na):
                                newline += " " + afters[j]
                        rinputs[i] = newline
            return rinputs
        else:
            return rinputs

    def close(self):
        try:
            self.bin.close()
        except:
            pass
