#ifndef TEST_BUILD_GA_H
#define TEST_BUILD_GA_H

#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

bool test_symmetric(const Eigen::MatrixXd& A);
bool test_antisymmetric(const Eigen::MatrixXd& A);
void test_majorana();
void test_kronecker();

#endif