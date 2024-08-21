import subprocess as sp


def workflow(L, seed, tol):
    stdout1 = sp.run(["sbatch", "build_syk.sh", str(L), str(seed)],
                     capture_output=True).stdout.decode("utf-8").strip()
    print(stdout1)
    jid1 = stdout1.split()[-1]
    stdout2 = sp.run(["sbatch",              f"--dependency=afterok:{jid1}", "solve_syk_powermethod.sh", str(L), str(seed)],
                     capture_output=True).stdout.decode("utf-8").strip()
    print(stdout2)
    jid2 = stdout2.split()[-1]
    stdout3 = sp.run(["sbatch",                  f"--dependency=afterok:{jid2}", "measure_obs.sh", str(L), str(seed), str(tol)],
                     capture_output=True).stdout.decode("utf-8").strip()
    print(stdout3)


Ls = [12]
seeds = [0]
tol = 0.001
for L in Ls:
    for seed in seeds:
        workflow(L, seed, tol)
