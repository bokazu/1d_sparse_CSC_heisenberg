#include "../all.h"

using namespace std;

void CSC_spin_operator(int j, int site_num, int tot_site_num, double *J,
                       int *row, double *mat_val, int &row_index,
                       int &col_ptr_val, double &szz)
{
    boost::dynamic_bitset<> ket_j(tot_site_num, j);
    boost::dynamic_bitset<> ket_j1(tot_site_num, j);
    bool bit_check0, bit_check1;
    // Point A
    if (site_num != tot_site_num - 1)
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(site_num + 1);
    }
    else
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(0);
    }

    if (bit_check0 == bit_check1)
    {
        // S^z_{i}S^z_{i+1}|1_{i+1} 1_{i}> or // S^z_{i}S^z_{i+1}|0_{i+1} 0_{i}>
        szz += 0.25 * J[site_num];
    }
    else
    {
        if (site_num != tot_site_num - 1)
        {
            ket_j1.flip(site_num + 1);
            ket_j1.flip(site_num);
        }
        else
        {
            ket_j1.flip(0);
            ket_j1.flip(site_num);
        }
        // Point C
        int i = ket_j1.to_ulong();

        // S^-_{i}S^+_{i+1} or S^+_{i}S^-_{i+1}
        row[row_index] = i;
        mat_val[row_index] = 0.5 * J[site_num];
        row_index++;
        col_ptr_val++;

        // S^z_{i}S^z_{i+1}|0_{i+1} 1_{i}>
        szz -= 0.25 * J[site_num];
    }
}

/*count non-zero elements*/
void spin_operator(int j, int site_num, int tot_site_num, double *J,
                   double &szz, int &mat_nonzero_elements)
{
    int diag_elements = 0;
    boost::dynamic_bitset<> ket_j(tot_site_num, j);
    boost::dynamic_bitset<> ket_j1(tot_site_num, j);
    bool bit_check0, bit_check1;
    // Point A
    if (site_num != tot_site_num - 1)
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(site_num + 1);
    }
    else
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(0);
    }

    if (bit_check0 == bit_check1)
    {
        // S^z_{i}S^z_{i+1}|1_{i+1} 1_{i}> or // S^z_{i}S^z_{i+1}|0_{i+1} 0_{i}>
        szz += 0.25 * J[site_num];
    }
    else
    {
        if (site_num != tot_site_num - 1)
        {
            ket_j1.flip(site_num + 1);
            ket_j1.flip(site_num);
        }
        else
        {
            ket_j1.flip(0);
            ket_j1.flip(site_num);
        }

        // S^-_{i}S^+_{i+1} or S^+_{i}S^-_{i+1}
        mat_nonzero_elements++;

        // S^z_{i}S^z_{i+1}|0_{i+1} 1_{i}>
        szz -= 0.25 * J[site_num];
    }
}