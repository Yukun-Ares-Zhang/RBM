#include <iostream>
#include <fstream>
#include <vector>
#include "Ising.h"

using namespace std;

#define TOTAL_TIME 10000
int main(){
    ofstream fout;
    fout.open("Metropolis.txt");
    srand((unsigned)time(nullptr));
    ising2D ising(100, 0, 2.);
    vector<double> m;
    double temp;
    int i;
    for (i = 0; i < TOTAL_TIME; i++)
    {
        temp = ising.magnetization();
        fout << i << "\t" << temp << endl;
        m.push_back(temp);
        ising.sweep();
    }
    temp = ising.magnetization();
    fout << i << "\t" << temp << endl;
    m.push_back(temp);
    fout.close();
    fout.open("correlation.txt");
    for(i = 0; i < 2000; i++)
    {
        double x1=0, x2=0, x3=0;
        for (int j = 0; j <= TOTAL_TIME-i; j++)
        {
            x1 += m.at(j) * m.at(j + i);
            x2 += m.at(j);
            x3 += m.at(j + i);
        }
        double x;
        x1 /= (TOTAL_TIME - i);
        x2 /= (TOTAL_TIME - i);
        x3 /= (TOTAL_TIME - i);
        x = x1 - x2 * x3;
        fout << i << "\t" << x << endl;
    }
    fout.close();
    return 0;
}