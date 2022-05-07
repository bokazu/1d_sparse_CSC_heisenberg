#include "../all.h"

using namespace std;

void ground_state_eigenvec(int mat_dim, int count,
                           double groundstate_eigen_value,
                           double *tri_diag_eigen_vec, double **u,
                           double *eigen_vec, double err)
{
    double lambda = groundstate_eigen_value;
    cout << "lambda = " << lambda << endl;

    for (int i = 0; i < count + 1; i++)
    {
        cblas_daxpy(mat_dim, tri_diag_eigen_vec[i], u[i], 1, eigen_vec, 1);
    }
    sdz(mat_dim, err, eigen_vec);
    for (int i = 0; i < mat_dim; i++)
    {
        cout << "eigen vector[" << i << "] = " << eigen_vec[i] << endl;
    }
}

void ground_eigenvec(int mat_dim, int count, double err,
                     double groundstate_eigen_value, double *alpha,
                     double *beta, double **u, double *eigen_vec)
{
    double *x = new double[mat_dim];
    vec_init(mat_dim, x);

    double lambda = groundstate_eigen_value;
    cout << "lambda = " << lambda << endl;
    x[0] = 1.0;
    x[1] = (lambda - alpha[0]) / beta[0];
    cout << "count = " << count << endl;
    for (int i = 1; i < count + 1; i++)
    {
        x[i + 1] =
            ((lambda - alpha[i]) * x[i] - beta[i - 1] * x[i - 1]) / beta[i];
    }
    // sdz(mat_dim, err, x);
    for (int i = 0; i < count + 1; i++)
    {
        cblas_daxpy(mat_dim, x[i], u[i], 1, eigen_vec, 1);
        cout << "eigen = " << endl;
        printvec(mat_dim, 5, eigen_vec);
        cout << "u = " << endl;
        printvec(mat_dim, 5, u[i]);
    }
    sdz(mat_dim, err, eigen_vec);
    for (int i = 0; i < mat_dim; i++)
    {
        cout << "eigen vector[" << i << "] = " << eigen_vec[i] << endl;
    }

    delete[] x;
}