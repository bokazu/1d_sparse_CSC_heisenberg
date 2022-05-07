#include "../all.h"

using namespace std;

void szz(int j, int site_num, int tot_site_num, double *J, int *row, int *col,
         double *mat_val, int &coo_index)
{
    // Point A2
    boost::dynamic_bitset<> ket_j(tot_site_num, j);
    bool bit_check0, bit_check1;

    // Point B2
    if (site_num == tot_site_num - 1)
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(0);
    }
    else
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(site_num + 1);
    }

    // Hamiltonianへの代入
    if (bit_check0 == bit_check1)
    {
        row[coo_index] = j;
        col[coo_index] = j;
        mat_val[coo_index] += 0.25 * J[site_num];
        cout << "/****************j=" << j << "**********************/" << endl;
        cout << "-0.25 * J[site_num] = " << 0.25 * J[site_num] << endl;
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
    }
    else
    {
        row[coo_index] = j;
        col[coo_index] = j;
        mat_val[coo_index] -= 0.25 * J[site_num];
        cout << "/****************j=" << j << "**********************/" << endl;
        cout << "-0.25 * J[site_num] = " << -0.25 * J[site_num] << endl;
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
    }
}

/*for the purpose to count matrix's non zero elements*/
void szz(
    int j, int site_num, int tot_site_num,
    int &mat_nonzero_elements)  // bra_iはsiteを変更するごとに手で変える必要あり
{
    // Point A2
    boost::dynamic_bitset<> ket_j(tot_site_num, j);
    bool bit_check0, bit_check1;

    // Point B2
    if (site_num == tot_site_num - 1)
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(0);
    }
    else
    {
        bit_check0 = ket_j.test(site_num);
        bit_check1 = ket_j.test(site_num + 1);
    }

    // Hamiltonianへの代入
    if (bit_check0 == bit_check1)
    {
        // H[j + j * dim] += 0.25 * J[site_num];
        // cout << "H[" << j << "+" << j << "*dim] = " << H[j + j * dim] <<
        // endl;
        mat_nonzero_elements++;
    }
    else
    {
        // H[j + j * dim] -= 0.25 * J[site_num];
        // cout << "H[" << j << "+" << j << "*dim] = " << H[j + j * dim] <<
        // endl;
        mat_nonzero_elements++;
    }
}