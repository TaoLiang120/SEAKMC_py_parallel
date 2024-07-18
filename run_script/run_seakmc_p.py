import time

from seakmc_p.input.Input import Settings
import seakmc_p.process.Preprocess as preseakmc
import seakmc_p.process.Process as runseakmc
import seakmc_p.process.Postprocess as postseakmc

def main():
    tic = time.time()
    inputf = "input.yaml"
    thissett = Settings.from_file(inputf)
    thissett.validate_input()

    seakmcdata, object_dict, Eground, thisRestart = preseakmc.preprocess(thissett)
    simulation_time = runseakmc.run_seakmc(thissett, seakmcdata, object_dict, Eground, thisRestart)
    postseakmc.postprocess(tic, thissett, object_dict, simulation_time)

if __name__ == '__main__':
    main()
