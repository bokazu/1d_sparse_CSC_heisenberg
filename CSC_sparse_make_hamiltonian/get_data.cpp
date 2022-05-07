#include "../all.h"

using namespace std;

void get_data(std::ifstream &if_M_H_Settingfile, int &tot_site_num,
              std::string &M_H_OutputFile_name, std::string &M_H_JsetFile_name,
              std::string &D_L_OutputFile_name, std::string &Boundary_Condition,
              int &precision)
{
    string ftmp;

    int fcount = 0;

    while (!if_M_H_Settingfile.eof())
    {
        stringstream ss;
        getline(if_M_H_Settingfile, ftmp);
        ss << ftmp;
        // cout << "ftmp=" << ftmp << endl ;
        if (fcount == 0)
        {
            ss >> tot_site_num;
            // std::cout << "tot_site_num=" << tot_site_num << endl;
        }
        else if (fcount == 1)
        {
            M_H_OutputFile_name = ftmp;
            // std::cout << "M_H_OutputFile_name = " << M_H_OutputFile_name <<
            // endl;
        }
        else if (fcount == 2)
        {
            M_H_JsetFile_name = ftmp;
            // std::cout << "M_H_JsetFile_name : " << M_H_JsetFile_name << endl;
        }
        else if (fcount == 3)
        {
            D_L_OutputFile_name = ftmp;
        }
        else if (fcount == 4)
        {
            Boundary_Condition = ftmp;
            // std::cout << "Boundary Condition : " << Boundary_Condition <<
            // endl;
        }
        else if (fcount == 5)
        {
            ss >> precision;
            // std::cout << "precision = " << precision << endl;
        }
        fcount += 1;
    }
    if_M_H_Settingfile.close();
}