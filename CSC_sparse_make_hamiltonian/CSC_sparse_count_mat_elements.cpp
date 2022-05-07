#include "../all.h"

using namespace std;

int CSC_sparse_count_mat_elements(int mat_dim, int tot_site_num,
                                  std::string M_H_JsetFile_name,
                                  std::string Boundary_Condition, double *J)
{
    int bond_num;
    int mat_nonzero_elements = 0;
    // int mat_diag_nonzero_elements = (tot_site_num - 1) * mat_dim;
    double szz;
    /*jset.txtからのbondごとの相互作用情報の取得*/
    /*bond数の取得*/

    if (Boundary_Condition == "y")
    {
        for (int j = 0; j < mat_dim; j++)
        {
            szz = 0.;
            for (int site_num = 0; site_num < tot_site_num; site_num++)
            {
                spin_operator(j, site_num, tot_site_num, J, szz,
                              mat_nonzero_elements);
            }
            if (szz != 0.0) mat_nonzero_elements++;
        }
    }
    else if (Boundary_Condition == "n")
    {
        for (int j = 0; j < mat_dim; j++)
        {
            szz = 0.;
            for (int site_num = 0; site_num < tot_site_num - 1; site_num++)
            {
                spin_operator(j, site_num, tot_site_num, J, szz,
                              mat_nonzero_elements);
            }
            if (szz != 0.0) mat_nonzero_elements++;
        }
    }
    else
    {
        cout << "ERROR : Maybe inputed other than \"y\" and \"n\" " << endl;
    }
    return mat_nonzero_elements;
}