#!/bin/bash

platform=$(uname)
if [ "${platform}" = "Darwin" ]; then
	CXX="clang++"
	INCLUDE_PATH="/Users/yitan/src/eigen-3.4.0"
else
	CXX="g++"
	INCLUDE_PATH="/n/home01/ytan/src/eigen"
	module load gcc/13.2.0-fasrc01
fi

rm -f build_syk
"${CXX}" -o build_syk build_syk.cc -fopenmp -I"${INCLUDE_PATH}" -std=c++20 -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations
