#include <iostream>
#include <fstream>
#include <vector>
#define TOTAL_TIME 10000
#define RELAX_TIME 2000
using namespace std;

int main(){
    ifstream infile;
    infile.open("Metropolis.txt", ios::in);
    if(!infile.is_open()){
        cerr << "Unable to open Metropolis.txt" << endl;
    }
    int t;
    double temp;
    vector<double> m;
    int count = 0;
    while (!infile.eof())
    {
        infile >> t >> temp;
        m.push_back(temp);
        count++;
    }
    
    ofstream fout;
    fout.open("correlation1.txt");
    for(int i = 0; i < 500; i++)
    {
        double x1=0, x2=0, x3=0;
        for (int j = RELAX_TIME; j <= TOTAL_TIME-i; j++)
        {
            x1 += m.at(j) * m.at(j + i);
            x2 += m.at(j);
            x3 += m.at(j + i);
        }
        double x;
        x1 /= (TOTAL_TIME - RELAX_TIME - i);
        x2 /= (TOTAL_TIME - RELAX_TIME - i);
        x3 /= (TOTAL_TIME - RELAX_TIME - i);
        x = x1 - x2 * x3;
        fout << i << "\t" << x << endl;
    }
    fout.close();
    return 0;
}