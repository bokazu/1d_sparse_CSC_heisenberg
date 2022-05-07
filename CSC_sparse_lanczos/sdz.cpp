#include "../all.h"

using namespace std;

void sdz(int mat_dim, double err, double* vec)
{
    double a = 1. / cblas_dnrm2(mat_dim, vec, 1);
    cblas_dscal(mat_dim, a, vec, 1);

    // Check norm
    double norm = cblas_dnrm2(mat_dim, vec, 1);
    double eps = abs(1.0 - norm);
    if (eps > 1.0e-15)
    {
        printf("\x1b[31m NORM ERROR\x1b[39m\n");
    }
}