#include <iostream>
#include <fstream>
#include <string>
#include "Ising.h"

using namespace std;
#define L 5
int main(){
    double T;
    ofstream fout;
    srand((unsigned)time(nullptr));
    string folder = "data/";
    string name1 = "_times_";
    string name2 = "_Metropolis.txt";
    string num = to_string(L);
    fout.open(folder + num + name1 + num + name2);
    if(!fout){
        cout << "Error: Can't open file." << endl;
    }
    for ( T = 0.2; T < 20.1; T+=0.2)
    {
        ising2D ising(L, 0, T);
        double E2 = 0, E = 0, m = 0, temp;
        int count = 0;
        for (int i = 0; i < 20000; i++)
        {
            ising.sweep();
            if(i<2000){
                continue;
            }
            m += fabs(ising.magnetization());
            temp = ising.Hamitonian();
            E += temp;
            E2 += temp * temp;
            count++;
        }
        E /= count;
        E2 /= count;
        m /= count;
        fout << T << "\t" << E << "\t" << m << "\t" << (E2 - E * E) / (L * L * T * T) << endl;
    }
    return 0;
}