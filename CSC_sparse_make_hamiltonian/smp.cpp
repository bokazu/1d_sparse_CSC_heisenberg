#include "../all.h"

using namespace std;

void smp(int j, int site_num, int tot_site_num, double *J, int *row, int *col,
         double *mat_val, int &coo_index)
{
    boost::dynamic_bitset<> ket_j(tot_site_num, j);
    boost::dynamic_bitset<> ket_j2(tot_site_num, j);
    bool bit_check0, bit_check1;

    // Point A1
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

    if ((bit_check0 == true) && (bit_check1 == false))
    {
        // Point B1
        if (site_num != tot_site_num - 1)
        {
            ket_j2.flip(site_num + 1);
            ket_j2.flip(site_num);
        }
        else
        {
            ket_j2.flip(0);
            ket_j2.flip(site_num);
        }
        // Point C
        int i = ket_j2.to_ulong();

        // Hamiltonianに代入
        // H[i + j * dim] += 0.5 * J[site_num];
        int row_num, col_num, val;
        row[coo_index] = i;
        col[coo_index] = j;
        mat_val[coo_index] = 0.5 * J[site_num];
        cout << "/****************j=" << j << "**********************/" << endl;
        cout << setw(6) << scientific << setprecision(5) << left << "row["
             << coo_index << "]"
             << "  " << setw(6) << scientific << setprecision(5) << left
             << "col[" << coo_index << "]"
             << "  " << setw(6) << scientific << setprecision(5) << left
             << "mat_val[" << coo_index << "]" << endl;
        cout << setw(6) << scientific << setprecision(5) << left
             << row[coo_index] << "  " << setw(6) << scientific
             << setprecision(5) << left << col[coo_index] << "  " << setw(6)
             << scientific << setprecision(5) << left << mat_val[coo_index]
             << endl;
        coo_index++;
    }
}

/*for the purpose to count matrix's non zero elements*/
void smp(int j, int site_num, int tot_site_num, int &mat_nonzero_elements)
{
    boost::dynamic_bitset<> ket_j(tot_site_num, j);
    boost::dynamic_bitset<> ket_j2(tot_site_num, j);
    bool bit_check0, bit_check1;

    // Point A1
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

    if ((bit_check0 == true) && (bit_check1 == false))
    {
        // Point B1
        if (site_num != tot_site_num - 1)
        {
            ket_j2.flip(site_num + 1);
            ket_j2.flip(site_num);
        }
        else
        {
            ket_j2.flip(0);
            ket_j2.flip(site_num);
        }
        // Point C
        int i = ket_j2.to_ulong();

        // Hamiltonianに代入
        // H[i + j * dim] += 0.5 * J[site_num];
        mat_nonzero_elements++;
    }
}