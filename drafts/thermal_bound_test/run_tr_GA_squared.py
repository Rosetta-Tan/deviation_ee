import subprocess
import sys, os

LAs = [6]
docker_python = ["docker", "exec", "-it", "0b53f3c7807fdb73f9f19aa08035361dff8c4b3581d50a9047c5e37e70ff963f", "python"]
seeds = range(0, 20)

for LA in LAs:
    for seed in seeds:
        # plain version
        # command = ['python', 'tr_GA_squared.py', '--LA', str(LA), '--seed', str(seed)]
        # dynamite version
        print(f'progress: LA={LA}, seed: [{seed}/{len(seeds)}]')
        command = docker_python + ['Coding/deviation_ee/drafts/thermal_bound_test/tr_GA_squared_dnm.py', '--LA', str(LA), '--seed', str(seed)]
        subprocess.run(command)