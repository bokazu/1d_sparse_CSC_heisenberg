#ifndef __MYlIB_H_
#define __MYlIB_H_

#include <boost/dynamic_bitset.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "cblas.h"
#include "lapacke.h"

/*make_hamiltonian*/
int CSC_sparse_count_mat_elements(int mat_dim, int tot_site_num,
                                  std::string M_H_JsetFile_name,
                                  std::string Boundary_Condition, double *J);
void get_data(std::ifstream &if_M_H_Settingfile, int &tot_site_num,
              int &tri_mat_dim, std::string &M_H_OutputFile_name,
              std::string &M_H_JsetFile_name, std::string &D_L_OutputFile_name,
              std::string &Boundary_Condition, int &precision);
/*for the purpose to count matrix's non zero elements*/
void spm(int j, int site_num, int tot_site_num, int &mat_nonzero_elements);
void smp(int j, int site_num, int tot_site_num, int &mat_nonzero_elements);
void szz(int j, int site_num, int tot_site_num, int &mat_nonzero_elements);
/*for the purpose to calculate matrix's non zero elements*/
void CSC_spin_operator(int j, int site_num, int tot_site_num, double *J,
                       int *row, double *mat_val, int &row_index,
                       int &col_ptr_val, double &szz);
void spin_operator(int j, int site_num, int tot_site_num, double *J,
                   double &szz, int &mat_nonzero_elements);

void CSC_sparse_make_hamiltonian(int mat_dim, int tot_site_num,
                                 std::string M_H_OutputFile_name, int precision,
                                 std::string Boundary_Condition, double *J,
                                 int *row, int *col_ptr, double *mat_val,
                                 int &nnz);

/*SPARSE LANCZOS*/
void CSC_sparse_lanczos(const int mat_dim, int nnz, int tri_mat_dim, int *row,
                        int *col_ptr, double *mat_val, double *eigen_value,
                        double *eigen_vec, std::string S_L_Outpufile_name);
void sdz(int mat_dim, double err, double *vec);
void gso(int n, int k, double **u);
void CSC_sparse_dgemv(int mat_dim, int nnz, double *v, int *row, int *col_ptr,
                      double *mat_val, double *u);
void ground_state_eigenvec(int mat_dim, int count,
                           double groundstate_eigen_value,
                           double *tri_diag_eigen_vec, double **u,
                           double *eigen_vec, double err);
void ground_eigenvec(int mat_dim, int count, double err,
                     double groundstate_eigen_value, double *alpha,
                     double *beta, double **u, double *eigen_vec);

/*共通な関数*/
void vec_init(int n, double *vec);
void vec_init(int n, int *vec);

/*行列出力用の関数テンプレート*/
template <typename T>
void printmat(int mat_dim, int precision, T *A)
{
    T mtmp;
    for (int row_num = 0; row_num < mat_dim; row_num++)
    {
        std::cout << "[";
        for (int col_num = 0; col_num < mat_dim; col_num++)
        {
            mtmp = A[col_num + mat_dim * row_num];
            std::cout << std::scientific << std::setprecision(precision)
                      << mtmp;
            if (col_num < mat_dim - 1)
            {
                std::cout << "  ";
            }
        }
        if (row_num < mat_dim - 1)
        {
            std::cout << "];" << std::endl;
        }
        else
        {
            std::cout << "]";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename T>
void fprintmat(std::ofstream &file, int mat_dim, int precision, T *A)
{
    // ofstream file(OutputFile_name, ios::app);
    T mtmp;
    for (int row_num = 0; row_num < mat_dim; row_num++)
    {
        for (int col_num = 0; col_num < mat_dim; col_num++)
        {
            mtmp = A[col_num + mat_dim * row_num];
            file << std::setw(5) << std::setprecision(precision) << std::left
                 << mtmp;
            if (col_num < mat_dim - 1)
            {
                file << "  ";
            }
        }
        if (row_num < mat_dim - 1)
        {
            file << std::endl;
        }
    }
    file << std::endl;
}

template <typename T>
void printvec(int vec_elements, int precision, T *v)
{
    T vtmp;
    std::cout << "[";
    for (int col_num = 0; col_num < vec_elements; col_num++)
    {
        vtmp = v[col_num];
        std::cout << std::setw(7) << std::scientific
                  << std::setprecision(precision) << std::left << vtmp;
        if (col_num < vec_elements - 1)
        {
            std::cout << "  ";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename T>
void fprintvec(std::ofstream &file, int vec_elements, int precision, T *v)
{
    T vtmp;
    for (int col_num = 0; col_num < vec_elements; col_num++)
    {
        vtmp = v[col_num];
        file << std::setw(17) << std::setprecision(precision) << vtmp;
        if (col_num < vec_elements - 1)
        {
            file << "  ";
        }
    }
    file << "\n" << std::endl;
}

/*ベクトルを列ベクトル表示する*/
template <typename T>
void fprintvec_col(std::ofstream &file, int vec_elements, int precision, T *v)
{
    T vtmp;
    for (int col_num = 0; col_num < vec_elements; col_num++)
    {
        vtmp = v[col_num];
        file << std::setw(17) << std::setprecision(precision) << vtmp
             << std::endl;
    }
}

template <typename T>
void print2dvec(int vec_elements, int precision, T **C)
{
    T vtmp;
    for (int row_num = 0; row_num < vec_elements; row_num++)
    {
        std::cout << "[";
        for (int col_num = 0; col_num < vec_elements; col_num++)
        {
            vtmp = C[row_num][col_num];
            std::cout << std::setw(7) << std::scientific
                      << std::setprecision(precision) << std::left << vtmp;
            if (col_num < vec_elements - 1)
            {
                std::cout << "  ";
            }
        }
        if (row_num < vec_elements - 1)
        {
            std::cout << "];" << std::endl;
        }
        else
        {
            std::cout << "]";
            std::cout << "]";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename T>
void fprint2dvec(std::ofstream &file, int vec_elements, int precision, T **C)
{
    T vtmp;
    for (int row_num = 0; row_num < vec_elements; row_num++)
    {
        for (int col_num = 0; col_num < vec_elements; col_num++)
        {
            vtmp = C[row_num][col_num];
            file << std::setw(17) << std::scientific
                 << std::setprecision(precision) << std::left << vtmp;
            if (col_num < vec_elements - 1)
            {
                file << "  ";
            }
        }
        if (row_num < vec_elements - 1)
        {
            file << std::endl;
        }
        else
        {
            file << std::endl;
        }
    }
    file << std::endl;
}

template <typename T>
void print_tri_diag_vec(int mat_dim, int precision, T *diag, T *sub_diag)
{
    T dtmp, subdtmp;
    for (int i = 0; i < mat_dim; i++)
    {
        for (int j = 0; j <= i - 2; j++)
        {
            std::cout << std::setw(7) << std::scientific
                      << std::setprecision(precision) << std::left << 0.0
                      << "  ";
        }
        if (i == 0)
        {
            std::cout << std::setw(7) << std::scientific
                      << std::setprecision(precision) << std::left << diag[i]
                      << "  " << sub_diag[i] << "  ";
        }
        else if (i == mat_dim - 1)
        {
            std::cout << std::setw(7) << std::scientific
                      << std::setprecision(precision) << std::left
                      << sub_diag[i] << "  " << diag[i] << "  ";
        }
        else
        {
            std::cout << std::setw(7) << std::scientific
                      << std::setprecision(precision) << std::left
                      << sub_diag[i - 1] << "  " << diag[i] << "  "
                      << sub_diag[i] << "  ";
        }
        for (int k = i + 2; k < mat_dim; k++)
        {
            if (k > 0)
            {
                std::cout << std::setw(7) << std::scientific
                          << std::setprecision(precision) << std::left << 0.0
                          << "  ";
            }
        }
        std::cout << std::endl;
    }
}

template <typename T>
void fprint_tri_diag_vec(std::ofstream &file, int mat_dim, int precision,
                         T *diag, T *sub_diag)
{
    T dtmp, subdtmp;
    for (int i = 0; i < mat_dim; i++)
    {
        for (int j = 0; j <= i - 2; j++)
        {
            file << std::setw(7) << std::scientific
                 << std::setprecision(precision) << std::left << 0.0 << "  ";
        }
        if (i == 0)
        {
            file << std::setw(7) << std::scientific
                 << std::setprecision(precision) << std::left << diag[i] << "  "
                 << sub_diag[i] << "  ";
        }
        else if (i == mat_dim - 1)
        {
            file << std::setw(7) << std::scientific
                 << std::setprecision(precision) << std::left << sub_diag[i - 1]
                 << "  " << diag[i] << "  ";
        }
        else
        {
            file << std::setw(7) << std::scientific
                 << std::setprecision(precision) << std::left << sub_diag[i - 1]
                 << "  " << diag[i] << "  " << sub_diag[i] << "  ";
        }
        for (int k = i + 2; k < mat_dim; k++)
        {
            if (k > 0)
            {
                file << std::setw(7) << std::scientific
                     << std::setprecision(precision) << std::left << 0.0
                     << "  ";
            }
        }
        file << std::endl;
    }
}
#endif