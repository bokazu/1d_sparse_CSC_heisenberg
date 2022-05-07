#include "all.h"

using namespace std;

void vec_init(int n, double* vec)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = 0.0;
    }
}

void vec_init(int n, int* vec)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = 0;
    }
}