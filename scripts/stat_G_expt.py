import os
import csv
import numpy as np

if __name__ == '__main__':
    data_folder = "/n/home01/ytan/scratch/deviation_ee/obs_syk/"
    Ls = [12, 14, 16, 18]
    seeds = list(range(20))
    tols = [0.1, 0.01, 0.001]
    expt_GA_data = np.zeros((len(Ls), len(tols), len(seeds)))
    for iL, L in enumerate(Ls):
        for iseed, seed in enumerate(seeds):
            for itol, tol in enumerate(tols):
                filepath = os.path.join(data_folder, f"expt_GA_L={L}_seed={seed}_tol={tol}.csv")
                with open(filepath, 'r') as file:
                    reader = csv.reader(file)
                    # header: L, seed, expt_GA
                    rows = list(reader)
                    for i in range(1, len(rows)):
                        if int(rows[i][0]) == L and int(rows[i][1]) == seed:
                            expt_GA = float(rows[i][2])
                            break
                expt_GA_data[iL, itol, iseed] = expt_GA
                
    
    with open(os.path.join(data_folder, "stat_expt_GA.csv"), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['L', 'tol', 'expt_GA'])
        for i in range(len(Ls)):
            for j in range(len(tols)):
                writer.writerow([Ls[i], tols[j], np.mean(expt_GA_data[i, j])])
                
    