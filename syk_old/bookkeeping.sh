#!/bin/bash

sed -i 's/scratch\/deviation_ee\//scratch\/deviation_ee\/old\//g' run_measure_obs.py
sed -i 's/deviation_ee\/syk\//deviation_ee\/syk_old\//g' run_measure_obs.py