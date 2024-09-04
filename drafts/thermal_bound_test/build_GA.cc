#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include "build_GA.h"
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

Eigen::MatrixXcd build_GA(const int L,
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
            if (cdnl == 6 || cdnl == 8) {
                complex<double> coeff = 24./(2*L*(2*L-1)*(2*L-2)*(2*L-3));
                for (int j = 0; j < cdnl; ++j) {
                    coeff *= complex<double>((2*L-j))/complex<double>(2*LA-j);
                }
                coeff *= cpls[i1] * cpls[i2];
                // MatrixXcd op = M[inds[i1][0]] * M[inds[i1][1]] * M[inds[i1][2]] * M[inds[i1][3]] * M[inds[i2][0]] * M[inds[i2][1]] * M[inds[i2][2]] * M[inds[i2][3]];
                MatrixXcd op(dim, dim);
                op.setIdentity();
                // for (int j = 0; j<4; ++j) {
                //     cerr << "inds[i1][j]: " << inds[i1][j] << "\n" << endl;
                //     cerr << "inds[i2][j]: " << inds[i2][j] << "\n" << endl;
                // }
                for (int j = 0; j < 4; ++j) {
                    assert(op.rows() == M[inds[i1][j]].rows() && "Matrix dimensions must match for multiplication");
                    MatrixXcd tmp = op * M[inds[i1][j]];
                    op = tmp;
                }
                for (int j = 0; j < 4; ++j) {
                    assert(op.rows() == M[inds[i2][j]].rows() && "Matrix dimensions must match for multiplication");
                    MatrixXcd tmp = op * M[inds[i2][j]];
                    op = tmp;
                }
                // cerr << "trace " << abs(op.trace()) << endl;
                assert(abs(op.trace()) < 1e-3 && "Matrix must be traceless");
                MatrixXcd GA_tmp = GA + coeff * op;
                GA = GA_tmp;
            }
        }
    }
    return GA;
}

Eigen::MatrixXcd majorana(int idx, int LA) {
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

    for (int j = spin_idx+1; j < LA; ++j) {
        int dim = pow(2, j+1);
        MatrixXcd m_temp = kroneckerProduct(sigma_0, m);
        m = m_temp;
        // cout << "step " << j << ": " << m.rows() << " " << m_temp.rows() << " " << dim << endl;
    }
    // cout << "dim: " << m.rows() << "\n" << endl;
    assert(m.rows() == pow(2, LA) && "Majorana matrix must have the right dimensions");
    return m;
}

vector<vector<int>> gen_inds(int LA) {
    int two_LA_choose_four = 2*LA*(2*LA-1)*(2*LA-2)*(2*LA-3)/24;
    vector<vector<int>> iterator(two_LA_choose_four, vector<int>(4, 0));
    int count = 0;
    for (int i = 0; i < 2*LA; ++i) {
        for (int j = i+1; j < 2*LA; ++j) {
            for (int k = j+1; k < 2*LA; ++k) {
                for (int l = k+1; l < 2*LA; ++l) {
                    iterator[count][0] = i;
                    iterator[count][1] = j;
                    iterator[count][2] = k;
                    iterator[count][3] = l;
                    count++;
                }
            }
        }
    }
    assert(count == two_LA_choose_four && "Iterator must have the right number of elements");
    return iterator;
}

int main(int argc, char* argv[]) {
    if(argc != 4) {
        cerr << "Usage: " << argv[0] << " L LA seed" << endl;
        return 1;
    }
    int L = stod(argv[1]);
    int LA = stoi(argv[2]);
    int seed = stoi(argv[3]);

    auto t0 = chrono::high_resolution_clock::now();

    test_count_unique();

    // cache Majoranas
    vector<Eigen::MatrixXcd> M;
    for (int i = 0; i < 2*LA; ++i) {
        M.push_back(majorana(i, LA));
    }

    // // Generate couplings
    // std::mt19937 gen(seed);
    // std::normal_distribution<double> dist(0., 1.);
    // int two_LA_choose_4 = (2*LA)*(2*LA-1)*(2*LA-2)*(2*LA-3)/24;
    // vector<double> cpls(two_LA_choose_4, 0.);
    // for (size_t i = 0; i < two_LA_choose_4; ++i) {
    //     cpls[i] = dist(gen);
    // }
    
    // load couplings
    string filename = "cpls_L=" + to_string(L) + "_LA=" + std::to_string(LA) + "_seed=" + std::to_string(seed) + ".txt";
    vector<double> cpls = loadCpls(filename);
    vector<vector<int>> inds = gen_inds(LA);

    // build GA
    cout << "Building GA" << endl;
    Eigen::MatrixXcd GA = build_GA(L, LA, inds, cpls, M);

    // Save real part
    string filename_real = "GA_real_L=" + to_string(L) + "_LA=" + std::to_string(LA) + "_seed=" + std::to_string(seed) + ".txt";
    saveMatrixToFile(GA.real(), filename_real);

    // Save imaginary part
    assert(test_antisymmetric(GA.imag()) && "Imaginary part of GA must be antisymmetric");
    string filename_imag = "GA_imag_L=" + std::to_string(L) + "_LA=" + std::to_string(LA) + "_seed=" + std::to_string(seed) + ".txt";
    saveMatrixToFile(GA.imag(), filename_imag);
    
    auto t1 = chrono::high_resolution_clock::now();
    cerr << "duration " << chrono::duration_cast<chrono::duration<double>>(t1 - t0) << endl;

    return 0;
}