#ifndef BUILD_GA_H
#define BUILD_GA_H

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;
using namespace std::complex_literals;
using namespace Eigen;

// routines
vector<double> loadCpls(const string& filename);
void saveMatrixToFile(const Eigen::MatrixXd& matrix, const string& filename);
vector<int> combine_inds(const vector<int>& v1, const vector<int>& v2);
int count_unique(const vector<int>& v);
Eigen::MatrixXcd build_GA(const int L,
                        const int LA,
                        const vector<vector<int>>& inds,
                        const vector<double>& cpls,
                        const vector<Eigen::MatrixXcd>& M);
Eigen::MatrixXcd majorana(int idx, int LA);
vector<vector<int>> gen_inds(int LA);

// tests
bool test_symmetric(const Eigen::MatrixXd& A);
bool test_antisymmetric(const Eigen::MatrixXd& A);
void test_majorana();
void test_kronecker();
void test_union_inds();
void test_count_unique();
void test_four_maj_term(const int L,
                        const int LA,
                        const vector<vector<int>>& inds,
                        const vector<double>& cpls,
                        const vector<Eigen::MatrixXcd>& M);

#endif