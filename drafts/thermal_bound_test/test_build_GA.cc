#include <iostream>
#include <vector>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include "build_GA.h"

bool test_symmetric(const Eigen::MatrixXd& A) {
    return A.isApprox(A.transpose());
}

bool test_antisymmetric(const Eigen::MatrixXd& A) {
    return A.isApprox(-A.transpose());
}

void test_majorana() {
    int LA = 6;
    vector<Eigen::MatrixXcd> M;
    for (int i = 0; i < 2*LA; ++i) {
        M.push_back(majorana(i, LA));
    }
    int i=0;
    int j=1;
    cout<< "anticommutation " << M[i] * M[j] + M[j] * M[i] << "\n" << endl;

    i=5;
    j=9;
    cout<< "anticommutation " << M[i] * M[j] + M[j] * M[i] << "\n" << endl;

    i=6;
    j=6;
    cout<< "anticommutation " << (M[i] * M[j] + M[j] * M[i]).diagonal() << "\n" << endl;
}

void test_four_maj_term(const int L,
                        const int LA,
                        const vector<vector<int>>& inds,
                        const vector<double>& cpls,
                        const vector<Eigen::MatrixXcd>& M) {
    int dim = pow(2, LA);  // LA is 13 at most, so won't overflow
    Eigen::MatrixXcd GA(dim, dim);
    GA.setZero();
    int two_LA_choose_four = 2*LA*(2*LA-1)*(2*LA-2)*(2*LA-3)/24;

    for (int i1 = 0; i1 < two_LA_choose_four; ++i1) {
        for (int i2 = 0; i2 < two_LA_choose_four; ++i2) {
            vector<int> union_inds = combine_inds(inds[i1], inds[i2]);
            int cdnl = count_unique(union_inds);
            if (cdnl == 4) {
                MatrixXcd op(dim, dim);
                op.setIdentity();
                for (int j = 0; j < 8; ++j) {
                    assert(op.cols() == M[union_inds[j]].rows() && "Matrix dimensions must match for multiplication");
                    MatrixXcd tmp = op * M[union_inds[j]];
                    op = tmp;
                }
            }
        }
    }
}

void test_kronecker() {
    MatrixXcd A(2, 2);
    A << 1, 2, 3, 4;
    MatrixXcd B(4, 4);
    B << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    int dim = pow(2, 3);
    MatrixXcd A_temp(dim, dim);
    A_temp = kroneckerProduct(A, B);
    A = A_temp;
    cout << A.rows() << " " << A.cols() << endl;   
}

void test_union_inds() {
    vector<vector<int>> inds = gen_inds(6);
    vector<int> u_inds = combine_inds(inds[0], inds[1]);
    for (int i = 0; i < u_inds.size(); ++i) {
        if (i != u_inds.size()-1) {
            cout << u_inds[i] << " ";
        } else {
            cout << u_inds[i] << endl;
        }
    }
}

void test_count_unique() {
    vector<int> inds = {0, 1, 2, 3, 0, 1, 2, 3};
    cout << count_unique(inds) << endl;
}