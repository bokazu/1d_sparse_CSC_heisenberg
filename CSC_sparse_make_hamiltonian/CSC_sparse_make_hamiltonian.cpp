/*M_H = Make_Hamiltonian*/
#include "../all.h"

using namespace std;

void CSC_sparse_make_hamiltonian(int mat_dim, int tot_site_num,
                                 std::string M_H_OutputFile_name, int precision,
                                 std::string Boundary_Condition, double *J,
                                 int *row, int *col_ptr, double *mat_val,
                                 int &nnz)
{
    int bond_num;
    int row_index = 0;
    int col_ptr_index = 0;
    int col_ptr_val = 0;
    double szz;

    if (Boundary_Condition == "y")
    {
        for (int j = 0; j < mat_dim; j++)
        {
            szz = 0.;
            col_ptr[col_ptr_index] = col_ptr_val;
            col_ptr_index++;

            for (int site_num = 0; site_num < tot_site_num; site_num++)
            {
                CSC_spin_operator(j, site_num, tot_site_num, J, row, mat_val,
                                  row_index, col_ptr_val, szz);
            }

            if (szz != 0.0)
            {
                mat_val[row_index] = szz;
                row[row_index] = j;
                row_index++;
                col_ptr_val++;
            }
        }
    }
    else if (Boundary_Condition == "n")
    {
        for (int j = 0; j < mat_dim; j++)
        {
            szz = 0.;
            col_ptr[col_ptr_index] = col_ptr_val;
            col_ptr_index++;
            for (int site_num = 0; site_num < tot_site_num - 1; site_num++)
            {
                CSC_spin_operator(j, site_num, tot_site_num, J, row, mat_val,
                                  row_index, col_ptr_val, szz);
            }

            if (szz != 0.0)
            {
                mat_val[row_index] = szz;
                row[row_index] = j;
                row_index++;
                col_ptr_val++;
            }
        }
    }
    else
    {
        cout << "ERROR : Maybe inputed other than \"y\" and \"n\" " << endl;
    }

    // col_ptrの最後尾の要素には非ゼロ要素数を代入する
    col_ptr[mat_dim] = nnz;

    // /*OUTPUT HAMILTONIAN*/
    ofstream M_H_Output(M_H_OutputFile_name);

    M_H_Output.close();
    delete[] J;
}