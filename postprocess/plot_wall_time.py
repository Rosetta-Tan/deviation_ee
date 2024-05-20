import os
import matplotlib.pyplot as plt
import csv
plt.style.use('../figures/norm.mplstyle')
obs_dir = '/n/home01/ytan/scratch/deviation_ee/obs_syk'

Ls = list(range(16, 27, 2))
wall_time_str = []
wall_time_seconds = []
with open(os.path.join(obs_dir, f'wall_time.csv'), 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        wall_time_str.append(row[1])
        wall_time_seconds.append(float(row[2]))

fig, ax = plt.subplots()
ax.plot(Ls, wall_time_seconds, '-o')
for i, txt in enumerate(wall_time_str):
    ax.annotate(txt, (Ls[i], wall_time_seconds[i]), textcoords="offset points", xytext=(0,10), ha='center')
ax.set_xlabel('L')
ax.set_xticks(Ls)
ax.set_ylabel('Wall time (s)')
fig.savefig('../figures/wall_time.pdf', bbox_inches='tight')


    