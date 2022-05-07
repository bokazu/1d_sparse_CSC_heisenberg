#include "../all.h"

using namespace std;

void gso(int n, int k, double** u)
{
    double cef;
    for (int try_num = 0; try_num < k - 1; try_num++)
    {
        cef = -1.0 * cblas_ddot(n, u[try_num], 1, u[k + 1], 1);
        cblas_daxpy(n, cef, u[try_num], 1, u[k + 1], 1);
    }
}