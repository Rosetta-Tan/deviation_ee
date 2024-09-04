import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, required=True, help='system size')
parser.add_argument('--LA_start', type=int, required=True, help='start of subsystem size')
args = parser.parse_args()
