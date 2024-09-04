platform = $(shell uname)
ifeq ($(platform), Darwin)
  CXX = clang++
else
  CXX = g++
endif

all: clean build_GA

build_GA: build_GA.cc test_build_GA.cc
	clang++ -o build_GA build_GA.cc test_build_GA.cc -I/Users/yitan/src/eigen-3.4.0 -std=c++20 -O3 -Wall -Wno-unused-variable

clean:
	rm -f build_GA