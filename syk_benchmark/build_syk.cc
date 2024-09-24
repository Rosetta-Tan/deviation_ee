#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <random>
#include <math.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include "build_syk.h"
using namespace std;
using namespace std::complex_literals;
using namespace Eigen;

vector<double> loadCpls(const string& filename) {
    vector<double> cpls;
    ifstream file(filename);
    if (file.is_open()) {
        double cpl;
        while (file >> cpl) {
            cpls.push_back(cpl);
        }
    }
    return cpls;
}

void saveMatrixToFile(const Eigen::MatrixXd& matrix, const string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << matrix;
    }
}

vector<int> combine_inds(const vector<int>& v1, const vector<int>& v2) {
    vector<int> union_inds;
    union_inds.reserve(v1.size() + v2.size());
    union_inds.insert(union_inds.end(), v1.begin(), v1.end());
    union_inds.insert(union_inds.end(), v2.begin(), v2.end()); 
    return union_inds;
}

int count_unique(const vector<int>& v) {
    vector<int> unique_v(v);
    sort(unique_v.begin(), unique_v.end());
    auto last = unique(unique_v.begin(), unique_v.end());
    return distance(unique_v.begin(), last);
}

Eigen::MatrixXcd build_syk(const int L,
                        const vector<vector<int>>& inds,
                        const vector<double>& cpls,
                        const vector<Eigen::MatrixXcd>& M) {
    int dim = pow(2, L);  // L is 13 at most, so won't overflow
    Eigen::MatrixXcd H(dim, dim);
    H.setZero();
    int two_L_choose_four = 2*L*(2*L-1)*(2*L-2)*(2*L-3)/24;
    auto t0 = chrono::high_resolution_clock::now();
    auto t1 = chrono::high_resolution_clock::now();
    MatrixXcd op(dim, dim);
    MatrixXcd op1(dim, dim);
    MatrixXcd op2(dim, dim);
    for (int i = 0; i < two_L_choose_four; ++i) {
        cerr << "Building term " << i << " of " << two_L_choose_four << endl;
        op1 = M[inds[i][0]] * M[inds[i][1]];
        t1 = chrono::high_resolution_clock::now();
        cerr << "duration " << chrono::duration_cast<chrono::duration<double>>(t1 - t0) << endl;
        op2 = M[inds[i][2]] * M[inds[i][3]];
        t1 = chrono::high_resolution_clock::now();
        cerr << "duration " << chrono::duration_cast<chrono::duration<double>>(t1 - t0) << endl;
        H.noalias() += op1 * op2;
        t1 = chrono::high_resolution_clock::now();
        cerr << "duration " << chrono::duration_cast<chrono::duration<double>>(t1 - t0) << endl;
    }
    complex<double> coeff = sqrt(24./(2*L*(2*L-1)*(2*L-2)*(2*L-3)));
    H = coeff * H;
    return H;
}

Eigen::MatrixXcd majorana(int idx, int L) {
    Eigen::MatrixXcd sigma_x{{0., 1.}, {1., 0.}};
    Eigen::MatrixXcd sigma_y{{0., -1i}, {1i, 0.}};
    Eigen::MatrixXcd sigma_z{{1., 0.}, {0., -1.}};
    Eigen::MatrixXcd sigma_0{{1., 0.}, {0., 1.}};
    Eigen::MatrixXcd m;
    int spin_idx = idx / 2;
    int parity = idx % 2;
    if (idx == 0) {
        m = sigma_x;
    } else if (idx == 1) {
        m = sigma_y;
    } else {
        m = sigma_z;
    }
    // cout << "step 0: " << m.rows() << endl;

    for (int k = 1; k < spin_idx; ++k) {
        int dim = pow(2, k+1);
        MatrixXcd m_temp = kroneckerProduct(sigma_z, m);
        m = m_temp;
        // cout << "step k= " << k << ": " << m.rows() << endl;
    }

    if (spin_idx != 0) {
        if (parity == 0) {
            int dim = pow(2, spin_idx+1);
            MatrixXcd m_temp(dim, dim);
            m_temp = kroneckerProduct(sigma_x, m);
            m = m_temp;
        } else {
            int dim = pow(2, spin_idx+1);
            MatrixXcd m_temp = kroneckerProduct(sigma_y, m);
            m = m_temp;
        }
        // cout << "step spin_idx= " << spin_idx << ": " << m.rows() << endl;
    }

    for (int j = spin_idx+1; j < L; ++j) {
        int dim = pow(2, j+1);
        MatrixXcd m_temp = kroneckerProduct(sigma_0, m);
        m = m_temp;
        // cout << "step " << j << ": " << m.rows() << " " << m_temp.rows() << " " << dim << endl;
    }
    // cout << "dim: " << m.rows() << "\n" << endl;
    assert(m.rows() == pow(2, L) && "Majorana matrix must have the right dimensions");
    return m;
}

vector<vector<int>> gen_inds(int L) {
    int two_L_choose_four = 2*L*(2*L-1)*(2*L-2)*(2*L-3)/24;
    vector<vector<int>> iterator(two_L_choose_four, vector<int>(4, 0));
    int count = 0;
    for (int i = 0; i < 2*L; ++i) {
        for (int j = i+1; j < 2*L; ++j) {
            for (int k = j+1; k < 2*L; ++k) {
                for (int l = k+1; l < 2*L; ++l) {
                    iterator[count][0] = i;
                    iterator[count][1] = j;
                    iterator[count][2] = k;
                    iterator[count][3] = l;
                    count++;
                }
            }
        }
    }
    assert(count == two_L_choose_four && "Iterator must have the right number of elements");
    return iterator;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        cerr << "Usage: " << argv[0] << " L seed" << endl;
        return 1;
    }
    int L = stoi(argv[1]);
    int seed = stoi(argv[2]);
    string data_dir = "data/";

    // cache Majoranas
    vector<Eigen::MatrixXcd> M;
    for (int i = 0; i < 2*L; ++i) {
        M.push_back(majorana(i, L));
    }
    
    // load couplings
    string filename = data_dir + "cpls_L=" + std::to_string(L) + "_seed=" + std::to_string(seed) + ".txt";
    vector<double> cpls = loadCpls(filename);
    vector<vector<int>> inds = gen_inds(L);

    // build SYK Hamiltonian
    cout << "Building SYK Hamiltonian" << endl;
    Eigen::MatrixXcd H = build_syk(L, inds, cpls, M);

    // Save real part
    string filename_real = data_dir + "H_real_L=" + std::to_string(L) + "_seed=" + std::to_string(seed) + ".txt";
    saveMatrixToFile(H.real(), filename_real);

    // Save imaginary part
    string filename_imag = data_dir + "H_imag_L=" + std::to_string(L) + "_seed=" + std::to_string(seed) + ".txt";
    saveMatrixToFile(H.imag(), filename_imag);

    return 0;
}
