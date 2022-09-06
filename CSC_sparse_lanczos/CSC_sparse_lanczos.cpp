#include "../all.h"

using namespace std;

void CSC_sparse_lanczos(const int mat_dim, int nnz, int tri_mat_dim, int *row,
                        int *col_ptr, double *mat_val, double *eigen_value,
                        double *eigen_vec,
                        std::string S_L_Outpufile_name)  // S_L = Sparse_Lanczos
{
    std::cout
        << "/******************************************************************"
           "****************SPARSE LANCZOS "
           "METHOD********************************************************/"
        << endl;

    ofstream of_S_L_Outputfile(S_L_Outpufile_name);
    if (!(of_S_L_Outputfile))
    {
        cerr << "Could not open the file -'" << S_L_Outpufile_name << "'"
             << endl;
    }

    int count = 0;
    double eps = 1.0;
    double err = 1.0e-16;
    bool err_checker = true;

    // setting Initial vector
    double **u = new double *[2];

    for (int i = 0; i < 2; i++)
    {
        u[i] = new double[mat_dim];
    }
    srand(time(NULL));
    for (int i = 0; i < mat_dim; i++)
    {
        u[0][i] = (double)rand() / RAND_MAX;
        u[1][i] = 0.0;
    }
    sdz(mat_dim, err, u[0]);
    cblas_dcopy(mat_dim, u[0], 1, eigen_vec, 1);

    // diagonal elements of tridiagonal matrix
    double *alpha = new double[tri_mat_dim];  // delete checked
    vec_init(tri_mat_dim, alpha);

    // sub diagonal elements of tridiagonal matrix
    double *beta = new double[tri_mat_dim];
    vec_init(tri_mat_dim, beta);  // delete checked

    /*arrays of eigen value*/
    double *eigen_value_even = new double[tri_mat_dim];  // delete checked
    double *eigen_value_odd = new double[tri_mat_dim];   // delete checked
    vec_init(tri_mat_dim, eigen_value_even);
    vec_init(tri_mat_dim, eigen_value_odd);

    /*diag = alpha , sub_diag = beta*/
    int tri_mat_dim2 = tri_mat_dim * tri_mat_dim;
    double *diag = new double[tri_mat_dim];                 // delete checked
    double *sub_diag = new double[tri_mat_dim];             // delete checked
    double *tri_diag_eigen_vec = new double[tri_mat_dim2];  // delete checked
    vec_init(tri_mat_dim, diag);
    vec_init(tri_mat_dim, sub_diag);
    vec_init(tri_mat_dim2, tri_diag_eigen_vec);

    for (int lanczos_step = 0; lanczos_step < 2; lanczos_step++)
    {
        // lanczos_step == 0 - > Caluculate Eigenvalue of Hamiltonian's ground
        // state
        if (lanczos_step == 0)
        {
            of_S_L_Outputfile << "Lanczos step = 0" << endl;
            for (int k = 0; k < tri_mat_dim; k++)
            {
                if (err_checker)
                {
                    count = k;
                    /************************************EVEN STEP
                     * START**********************************/
                    if (k % 2 == 0)
                    {
                        if (k == mat_dim - 1)
                        {
                            cblas_dscal(mat_dim, -beta[k - 1], u[1], 1);
                            CSC_sparse_dgemv(mat_dim, nnz, u[1], row, col_ptr,
                                             mat_val, u[0]);
                            alpha[k] = cblas_ddot(mat_dim, u[1], 1, u[0], 1);

                            // calculate eigenvalue
                            vec_init(tri_mat_dim, diag);
                            vec_init(tri_mat_dim, sub_diag);
                            cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                            cblas_dcopy(tri_mat_dim, beta, 1, sub_diag, 1);

                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', mat_dim, diag,
                                          sub_diag, tri_diag_eigen_vec,
                                          mat_dim);
                            cblas_dcopy(mat_dim, diag, 1, eigen_value_even, 1);
                        }
                        else
                        {
                            if (k == 0)
                            {
                                of_S_L_Outputfile << "u[0] = " << endl;
                                fprintvec(of_S_L_Outputfile, mat_dim, 5, u[0]);
                                CSC_sparse_dgemv(mat_dim, nnz, u[1], row,
                                                 col_ptr, mat_val, u[0]);
                                alpha[k] =
                                    cblas_ddot(mat_dim, u[1], 1, u[0], 1);
                                cblas_daxpy(mat_dim, -alpha[k], u[0], 1, u[1],
                                            1);
                                beta[k] = cblas_dnrm2(mat_dim, u[1], 1);
                                cblas_dscal(mat_dim, 1. / beta[k], u[1], 1);
                                of_S_L_Outputfile << "u[1] = " << endl;
                                fprintvec(of_S_L_Outputfile, mat_dim, 5, u[1]);

                                // calculate eigenvalue
                                vec_init(tri_mat_dim, diag);
                                vec_init(tri_mat_dim, sub_diag);
                                cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                                cblas_dcopy(tri_mat_dim, beta, 1, sub_diag, 1);
                                int info = LAPACKE_dstev(
                                    LAPACK_COL_MAJOR, 'V', k + 2, diag,
                                    sub_diag, tri_diag_eigen_vec, k + 2);
                                if (info != 0)
                                {
                                    std::cout << "At k = " << k
                                              << " , LAPACKE_detev errored."
                                              << endl;
                                }
                                cblas_dcopy(tri_mat_dim, diag, 1,
                                            eigen_value_even, 1);
                            }
                            else
                            {
                                cblas_dscal(mat_dim, -beta[k - 1], u[1], 1);
                                CSC_sparse_dgemv(mat_dim, nnz, u[1], row,
                                                 col_ptr, mat_val, u[0]);
                                alpha[k] =
                                    cblas_ddot(mat_dim, u[1], 1, u[0], 1);
                                cblas_daxpy(mat_dim, -alpha[k], u[0], 1, u[1],
                                            1);
                                beta[k] = cblas_dnrm2(mat_dim, u[1], 1);
                                cblas_dscal(mat_dim, 1. / beta[k], u[1], 1);
                                of_S_L_Outputfile << "u[" << k + 1
                                                  << "] = " << endl;
                                fprintvec(of_S_L_Outputfile, mat_dim, 5, u[1]);
                                // calculate eigenvalue
                                vec_init(tri_mat_dim, diag);
                                vec_init(tri_mat_dim, sub_diag);
                                cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                                cblas_dcopy(tri_mat_dim, beta, 1, sub_diag, 1);
                                if (k < tri_mat_dim - 1)
                                {
                                    int info = LAPACKE_dstev(
                                        LAPACK_COL_MAJOR, 'V', k + 2, diag,
                                        sub_diag, tri_diag_eigen_vec, k + 2);
                                    if (info != 0)
                                    {
                                        std::cout << "At k = " << k
                                                  << " , LAPACKE_detev errored."
                                                  << endl;
                                    }
                                }
                                else
                                {
                                    int info = LAPACKE_dstev(
                                        LAPACK_COL_MAJOR, 'V', k, diag,
                                        sub_diag, tri_diag_eigen_vec, k);
                                    if (info != 0)
                                    {
                                        std::cout << "At k = " << k
                                                  << " , LAPACKE_detev errored."
                                                  << endl;
                                    }
                                }
                                cblas_dcopy(tri_mat_dim, diag, 1,
                                            eigen_value_even, 1);
                            }
                        }
                    }
                    /***************************EVEN STEP
                     * END**********************************/
                    /***************************ODD STEP
                     * STRT*********************************/
                    else
                    {
                        if (k == mat_dim - 1)
                        {
                            cblas_dscal(mat_dim, -beta[k - 1], u[0], 1);
                            CSC_sparse_dgemv(mat_dim, nnz, u[0], row, col_ptr,
                                             mat_val, u[1]);
                            alpha[k] = cblas_ddot(mat_dim, u[0], 1, u[1], 1);

                            // calculate eigenvalue
                            vec_init(tri_mat_dim, diag);
                            vec_init(tri_mat_dim, sub_diag);
                            cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                            cblas_dcopy(tri_mat_dim, beta, 1, sub_diag, 1);
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', mat_dim, diag,
                                          sub_diag, tri_diag_eigen_vec,
                                          mat_dim);
                            cblas_dcopy(mat_dim, diag, 1, eigen_value_odd, 1);
                        }
                        else
                        {
                            cblas_dscal(mat_dim, -beta[k - 1], u[0], 1);
                            CSC_sparse_dgemv(mat_dim, nnz, u[0], row, col_ptr,
                                             mat_val, u[1]);
                            alpha[k] = cblas_ddot(mat_dim, u[0], 1, u[1], 1);
                            cblas_daxpy(mat_dim, -alpha[k], u[1], 1, u[0], 1);
                            beta[k] = cblas_dnrm2(mat_dim, u[0], 1);
                            cblas_dscal(mat_dim, 1. / beta[k], u[0], 1);
                            of_S_L_Outputfile << "u[" << k + 1
                                              << "] = " << endl;
                            fprintvec(of_S_L_Outputfile, mat_dim, 5, u[0]);

                            // calculate eigenvalue
                            vec_init(tri_mat_dim, diag);
                            vec_init(tri_mat_dim, sub_diag);
                            cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                            cblas_dcopy(tri_mat_dim, beta, 1, sub_diag, 1);
                            if (k < tri_mat_dim - 1)
                            {
                                int info = LAPACKE_dstev(
                                    LAPACK_COL_MAJOR, 'V', k + 2, diag,
                                    sub_diag, tri_diag_eigen_vec, k + 2);
                                if (info != 0)
                                {
                                    std::cout << "At k = " << k
                                              << " , LAPACKE_detev errored."
                                              << endl;
                                }
                            }
                            else
                            {
                                int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'V',
                                                         k, diag, sub_diag,
                                                         tri_diag_eigen_vec, k);
                                if (info != 0)
                                {
                                    std::cout << "At k = " << k
                                              << " , LAPACKE_detev errored."
                                              << endl;
                                }
                            }
                            cblas_dcopy(tri_mat_dim, diag, 1, eigen_value_odd,
                                        1);
                        }
                    }

                    if (k > 0)
                    {
                        eps = abs(eigen_value_even[0] - eigen_value_odd[0]);
                        if (eps > err)
                        {
                            err_checker = true;  // continue calculation
                        }
                        else
                        {
                            err_checker = false;
                        }
                    }
                }
                else
                {
                    cout << "Break At Count : " << count << endl;
                    break;
                }
            }
            if (count % 2 == 0)
            {
                cblas_dcopy(tri_mat_dim, eigen_value_even, 1, eigen_value, 1);
            }
            else
            {
                cblas_dcopy(tri_mat_dim, eigen_value_odd, 1, eigen_value, 1);
            }
            cout << "FINISH CALCULATING EIGEN VALUE" << endl;
        }
        else  // Case Lanczos Step = 1 , calculate eigen vector
        {
            // of_S_L_Outputfile << "Lanczos step = 1" << endl;
            // cout << "START CALCULATION OF EIGEN VECTOR" << endl;
            // for (int k = 0; k < count + 2; k++)
            // {
            //     if (k % 2 == 0)
            //     {
            //         if (k == mat_dim - 1)
            //         {
            //             // Calculate Eigen Vector
            //             cblas_daxpy(mat_dim, tri_diag_eigen_vec[k], u[0], 1,
            //                         eigen_vec, 1);
            //             // Caluculate alpha
            //             cblas_dscal(mat_dim, -beta[k - 1], u[1], 1);
            //             CSC_sparse_dgemv(mat_dim, nnz, u[1], row, col_ptr,
            //                              mat_val, u[0]);
            //         }
            //         else
            //         {
            //             if (k == 0)
            //             {
            //                 vec_init(mat_dim, u[1]);
            //                 vec_init(mat_dim, u[0]);
            //                 cblas_dcopy(mat_dim, eigen_vec, 1, u[0], 1);
            //                 cblas_dscal(mat_dim, tri_diag_eigen_vec[k],
            //                             eigen_vec, 1);
            //                 // Calculate u[i+1]
            //                 CSC_sparse_dgemv(mat_dim, nnz, u[1], row,
            //                 col_ptr,
            //                                  mat_val, u[0]);
            //                 cblas_daxpy(mat_dim, -alpha[k], u[0], 1, u[1],
            //                 1); cblas_dscal(mat_dim, 1. / beta[k], u[1], 1);
            //             }
            //             else
            //             {
            //                 // Calculate Eigen Vector
            //                 cblas_daxpy(mat_dim, tri_diag_eigen_vec[k], u[0],
            //                 1,
            //                             eigen_vec, 1);
            //                 // Caluculate u[i+1]
            //                 cblas_dscal(mat_dim, -beta[k - 1], u[1], 1);
            //                 CSC_sparse_dgemv(mat_dim, nnz, u[1], row,
            //                 col_ptr,
            //                                  mat_val, u[0]);
            //                 cblas_daxpy(mat_dim, -alpha[k], u[0], 1, u[1],
            //                 1); cblas_dscal(mat_dim, 1. / beta[k], u[1], 1);
            //             }
            //         }
            //     }
            //     else
            //     {
            //         if (k == mat_dim - 1)
            //         {
            //             // Calculate Eigen Vector
            //             cblas_daxpy(mat_dim, tri_diag_eigen_vec[k], u[1], 1,
            //                         eigen_vec, 1);
            //             of_S_L_Outputfile << "x(at" << k << ") = " << endl;
            //             fprintvec(of_S_L_Outputfile, mat_dim, 5, eigen_vec);
            //             // Caluculate alpha
            //             cblas_dscal(mat_dim, -beta[k - 1], u[0], 1);
            //             CSC_sparse_dgemv(mat_dim, nnz, u[0], row, col_ptr,
            //                              mat_val, u[1]);
            //             // alpha[k] = cblas_ddot(mat_dim, u[0], 1, u[1], 1);
            //         }
            //         else
            //         {  // Calculate Eigen Vector
            //             cblas_daxpy(mat_dim, tri_diag_eigen_vec[k], u[1], 1,
            //                         eigen_vec, 1);
            //             of_S_L_Outputfile << "x(at" << k << ") = " << endl;
            //             fprintvec(of_S_L_Outputfile, mat_dim, 5, eigen_vec);
            //             // Caluculate alpha,beta,u[i+1]
            //             cblas_dscal(mat_dim, -beta[k - 1], u[0], 1);
            //             CSC_sparse_dgemv(mat_dim, nnz, u[0], row, col_ptr,
            //                              mat_val, u[1]);
            //             // alpha[k] = cblas_ddot(mat_dim, u[0], 1, u[1], 1);
            //             cblas_daxpy(mat_dim, -alpha[k], u[1], 1, u[0], 1);
            //             // beta[k] = cblas_dnrm2(mat_dim, u[0], 1);
            //             cblas_dscal(mat_dim, 1. / beta[k], u[0], 1);
            //             of_S_L_Outputfile << "u[" << k + 1 << "] = " << endl;
            //             fprintvec(of_S_L_Outputfile, mat_dim, 5, u[0]);
            //         }
            //     }
            // }
        }
    }
    // sdz(mat_dim, err, eigen_vec);
    std::cout << "END" << endl;

    std::cout << "eigen value = " << eigen_value[0] << endl;
    // /**OUTPUT TO FILE**/
    // of_S_L_Outputfile << "Break at count = " << count << endl;
    // of_S_L_Outputfile << "1. Ground State of EIGEN VALUES" << endl;
    // of_S_L_Outputfile << eigen_value[0] << endl;
    // fprintvec(of_S_L_Outputfile, tri_mat_dim, 5, eigen_value);
    // of_S_L_Outputfile << endl;
    // of_S_L_Outputfile << "2. EIGEN VECTOR OF GROUND STATE" << endl;
    // fprintvec(of_S_L_Outputfile, mat_dim, 5, eigen_vec);
    // of_S_L_Outputfile << endl;

    /*release memory*/
    for (int i = 0; i < 2; i++)
    {
        delete[] u[i];
    }

    delete[] u;
    delete[] alpha;
    delete[] beta;
    delete[] eigen_value_even;
    delete[] eigen_value_odd;
    delete[] tri_diag_eigen_vec;
    delete[] diag;
    delete[] sub_diag;
}
