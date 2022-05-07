#include "../all.h"

using namespace std;

void sparse_dgemv(int mat_dim, int mat_elements, double *v, int *row, int *col,
                  double *mat_val, double *u)
{
    int row_num, col_num;
    for (int i = 0; i < mat_elements; i++)
    {
        row_num = row[i];
        col_num = col[i];
        v[row_num] += mat_val[i] * u[col_num];
    }
}