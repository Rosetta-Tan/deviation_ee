#!/bin/bash

docker run --runtime=nvidia -v /home/yitan/Coding/deviation_ee:/home/dnm gdmeyer/dynamite:latest-cuda python syk/build_syk.py --L 12 --seed 0 --dirc ./ --gpu 1