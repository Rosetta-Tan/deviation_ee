import subprocess as sp
import sys

def workflow(L, seed, tol):
    stdout1 = sp.run(["sbatch", "build_syk.sh", str(L), str(seed)],
                     capture_output=True).stdout.decode("utf-8").strip()
    print(stdout1)
    jid1 = stdout1.split()[-1]
    stdout2 = sp.run(["sbatch", f"--dependency=afterok:{jid1}", "solve_syk_powermethod.sh", str(L), str(seed), str(tol)], capture_output=True).stdout.decode("utf-8").strip()
    print(stdout2)
    jid2 = stdout2.split()[-1]
    stdout3 = sp.run(["sbatch", f"--dependency=afterok:{jid2}", "measure_obs.sh", str(L), str(seed), str(tol)],
                     capture_output=True).stdout.decode("utf-8").strip()
    print(stdout3)

def submit_measure_tmp(L, seed, tol):
    stdout = sp.run(["sbatch", "measure_obs.sh", str(L), str(seed), str(tol)], capture_output=True).stdout.decode("utf-8").strip()
    print(stdout)

def resolve_dependency(thisjob, nextjob):
    # 1. check if this job is done
    # 2. if not, wait for 5 minutes
    pass

if __name__ == "__main__":
    Ls = [24]
    seeds = [i for i in range(20)]
    tols = [0.1, 0.01, 0.001]

    for L in Ls:
        for seed in seeds:
            for tol in tols:
                workflow(L, seed, tol)
                # submit_measure_tmp(L, seed, tol)