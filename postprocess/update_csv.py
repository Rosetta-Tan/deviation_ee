import csv, os
import numpy as np

# grid
# Ls = list(range(21, 25))
# seeds = list(range(0, 100))

# supplemental
# Ls = [15]
# seeds = [28, 58, 83]
# Ls = [22]
# seeds = [12, 37, 46, 62, 80, 81, 86, 92, 98]
Ls = [26]
seeds = list(range(0, 20))


def initialize_csv(L_range, seed_range, mode='w'):
    # The file to write to
    filename = 'extrm_eigvals.csv'

    # Open the file and write the headers and rows
    with open(filename, mode=mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        if mode == 'w':
            writer.writerow(['L', 'seed', 'extrm_eigval'])
        
        # Iterate through each L and seed combination and write the rows
        for L in L_range:
            for seed in seed_range:
                # Placeholder for extremal eigenvalue (you can fill this in)
                extremal_eigenvalue = 'PLACEHOLDER'
                writer.writerow([L, seed, extremal_eigenvalue])

    print(f'CSV file {filename} has been created.')

def update_csv(L, seed, new_value):
    # Open the existing CSV file
    filename = 'extrm_eigvals.csv'
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Find the row that matches the given L and seed
    for i in range(1, len(rows)):
        if int(rows[i][0]) == L and int(rows[i][1]) == seed:
            # Update the extremal eigenvalue with the new value
            rows[i][2] = new_value
            break

    # Write the updated rows back to the CSV file
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f'CSV file {filename} has been updated with the new value.')

if __name__ == '__main__':
    # initialize_csv(Ls, seeds, mode='a')
    for L in Ls:
        for seed in seeds:
            filename = f'/n/home01/ytan/scratch/deviation_ee/output/solve_extrm_eigvals/evals_L={L}_seed={seed}.npz'
            if not os.path.isfile(filename):
                continue
            extremal_eigenvalue = np.load(filename)['evals'][0]
            update_csv(L, seed, extremal_eigenvalue)