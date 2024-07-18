import time

import numpy as np

from seakmc_p.core.data import ActiveVolumeSPS
from seakmc_p.core.util import mats_angle
from seakmc_p.spsearch.SPSearch import Dimer
from seakmc_p.spsearch.SaddlePoints import SaddlePoint

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def generate_VN(spsearchsett, thisVNS, nactive, SNC=False, dmAV=None):
    isvalid = False
    thisiter = 0
    while not isvalid:
        if SNC:
            VN = np.random.rand(3 * nactive) - 0.5
            isel = np.random.randint(0, high=3 * nactive)
            VN += dmAV.eigvec.T[isel]
            VN = VN.reshape([3, nactive])
        else:
            VN = np.random.rand(3, nactive) - 0.5
        if spsearchsett["HandleVN"]["CheckAng4Init"]:
            anglemin = 180.0
            for i in range(len(thisVNS)):
                angle = mats_angle(VN, thisVNS[i], Flatten=True)
                if angle < anglemin: anglemin = angle

            if (anglemin < spsearchsett["HandleVN"]["AngTol4Init"] and
                    thisiter <= spsearchsett["HandleVN"]["MaxIter4Init"]):
                isvalid = False
            else:
                isvalid = True
            thisiter += 1
        else:
            isvalid = True
    return VN


def spsearch_search_single(nproc_task, thiscolor, comm_split,
                           idav, thissett, thisAV, thisSOPs, dynmatAV, SNC, CalPref, thisSPS, Pre_Disps,
                           idsps, thisVNS, object_dict):
    ticd = time.time()
    force_evaluator = object_dict['force_evaluator']

    thisAVd = ActiveVolumeSPS.from_activevolume(idsps, thisAV)
    if comm_split.Get_rank() == 0:
        thisVN = generate_VN(thissett.spsearch, thisVNS, thisAVd.nactive, SNC=SNC, dmAV=dynmatAV)
    else:
        thisVN = None
    if comm_split.Get_size() > 1: thisVN = comm_split.bcast(thisVN, root=0)
    if thissett.spsearch["Method"].upper() == "DIMER":
        thispredisp = []
        if idsps < len(Pre_Disps):
            thispredisp = Pre_Disps[idsps]
            Pre_Disps[idsps] = np.array([])
        thisspsearch = Dimer(idav, idsps, thisAVd, thissett, thiscolor, force_evaluator,
                             SNC=SNC, dmAV=dynmatAV, pre_disps=thispredisp, apply_mass=thissett.spsearch["ApplyMass"],
                             comm=comm_split)

        thisspsearch.dimer_init(thisVN)
        thisspsearch.dimer_search(thisSPS)
        if thissett.spsearch["SearchBuffer"]:
            thisspsearch.dimer_re_search(thisSPS, nactive=thisAV.nactive + thisAV.nbuffer)
        thisspsearch.dimer_finalize()  ##local relaxation
        if CalPref and thisspsearch.ISVALID:
            toDel = thisspsearch.is_to_be_delete()
            if not toDel:
                thisspsearch.calculate_prefactor()
        thisspsearch.dimer_finish()  ##

    thisSP = SaddlePoint(idav, idsps, idsps + 1, thisspsearch.BARR, thisspsearch.PREF, thisspsearch.EBIAS,
                         thisspsearch.ISCONNECT,
                         thisspsearch.XDISP, thisspsearch.DMAG, thisspsearch.DMAT, thisspsearch.DVEC,
                         thisspsearch.FXDISP, thisspsearch.FDMAG, thisspsearch.ISVALID,
                         iters=thisspsearch.iter, ntrans=thisspsearch.NTSITR, emax=thisspsearch.EDIFF_MAX,
                         rdcut=thissett.spsearch["R2Dmax4SPAtom"], dcut=thissett.spsearch["DCut4SPAtom"],
                         dyncut=thissett.spsearch["DynCut4SPAtom"],
                         tol=thissett.system["Tolerance"])

    tocd = time.time()
    return thisSP, thisVN, tocd - ticd, thisspsearch.NTSITR
