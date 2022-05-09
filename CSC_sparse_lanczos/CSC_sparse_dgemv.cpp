#include "../all.h"

using namespace std;

void CSC_sparse_dgemv(int mat_dim, int nnz, double *v, int *row, int *col_ptr,
                      double *mat_val, double *u)
{
    for (int i = 0; i < mat_dim; i++)
    {
        for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
        {
            v[i] += mat_val[j] * u[row[j]];
        }
    }
}