#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
struct erg{
    double energy;
    int num;
    erg(double energy_i = 0, int num_i = 0){
        energy = energy_i;
        num = num_i;
    }
};

class ising2D_rigor{
private:
    int m_L, m_N, m_XNN, m_YNN;
    int* m_s;
    double m_J;
    vector<erg> Hu;
public:
    ising2D_rigor(int L);  
    double Hamitonian();
    int add_erg(double E);
    void cprint_vector_erg();
    void print_m_s();
    double get_mean_erg(double T);
    double get_mean_square_erg(double T);
    double get_specific_heat(double T);
    void fprint_vector_erg(string str);
};

ising2D_rigor::ising2D_rigor(int L): m_L(L), m_N(L*L), m_XNN(1), m_YNN(L), m_J(1.){
    m_s = new int[m_N];
    for(int i=0;i<m_N;i++){
        m_s[i] = 1;
    }
    double E;
    while(m_s[0] == 1){
        E = Hamitonian();
        add_erg(E);
        if(m_s[m_N-1] == 1){
            m_s[m_N-1] = -1; 
        }
        else{
            for(int i = m_N - 1;i >= 0; i--){
                if(m_s[i] == 1){
                    m_s[i] = -1;
                    for(int j=i+1; j<m_N; j++) m_s[j] = 1;
                    break;
                }
            }
        }
    }
    for(auto i = Hu.begin(); i < Hu.end(); i++){
        (*i).num *= 2;
    }
}
double ising2D_rigor::Hamitonian(){
    double hamitonian=0;
    for (int i = 0; i < m_N; i++)
    {
        int nn, sum = 0;
        if ((nn = i + m_XNN) >= m_N)
            nn -= m_N;
        sum += m_s[nn];
        if ((nn = i - m_XNN) < 0)
            nn += m_N;
        sum += m_s[nn];
        if ((nn = i + m_YNN) >= m_N)
            nn -= m_N;
        sum += m_s[nn];
        if ((nn = i - m_YNN) < 0)
            nn += m_N;
        sum += m_s[nn];
        hamitonian += m_s[i] * sum;
    }
    hamitonian *= (-0.5);
    return hamitonian;    
}
int ising2D_rigor::add_erg(double E)
{
    for (auto i = Hu.begin(); i < Hu.end(); i++)
    {
        if ((*i).energy == E)
        {
            (*i).num++;
            return 1;
        }
    }
    erg temp;
    temp.energy = E;
    temp.num = 1;
    Hu.push_back(temp);
    return 1;
}
void ising2D_rigor::cprint_vector_erg()
{
    cout << "--------------------------" << endl;
    for (auto i = Hu.begin(); i < Hu.end(); i++)
    {
        cout << "Energy=" << (*i).energy << ", num=" << (*i).num << endl;
    }
    cout << "--------------------------" << endl;
}
void ising2D_rigor::print_m_s(){
    cout << "(";
    for (int i = 0; i < m_N; i++){
        cout << m_s[i];
        if(i!= m_N-1)
            cout << ",";
    }
    cout << ")" << endl;
}

double ising2D_rigor::get_mean_erg(double T){
    double E = 0, Z = 0;
    for (auto i = Hu.begin(); i < Hu.end(); i++)
    {
        E += (*i).num * (*i).energy * exp(-(*i).energy/T);
        Z += (*i).num * exp(-(*i).energy/T);
    }
    E /= Z;
    return E;
}
double ising2D_rigor::get_mean_square_erg(double T){
    double E2 = 0, Z = 0;
    for (auto i = Hu.begin(); i < Hu.end(); i++)
    {
        E2 += (*i).num * (*i).energy * (*i).energy * exp(-(*i).energy/T);
        Z += (*i).num * exp(-(*i).energy/T);
    }
    E2 /= Z;
    return E2;
}
double ising2D_rigor::get_specific_heat(double T){
    double E = 0, E2 = 0, Z = 0;
    for (auto i = Hu.begin(); i < Hu.end(); i++)
    {
        E += (*i).num * (*i).energy * exp(-(*i).energy/T);
        E2 += (*i).num * (*i).energy * (*i).energy * exp(-(*i).energy/T);
        Z += (*i).num * exp(-(*i).energy/T);
    }
    E /= Z;
    E2 /= Z;
    double c = (E2 - E * E) / (m_N * T * T);
    return c;
}
void ising2D_rigor::fprint_vector_erg(string str){
    ofstream fout;
    fout.open(str);
    if(!fout){
        cout << "Cannot open file: " << str << endl;
        return;
    }
    for (auto i = Hu.begin(); i < Hu.end(); i++)
    {
        fout << setw(10) << (*i).energy << "\t" << setw(10) << (*i).num << endl;
    }
    fout.close();
}
inline double specific_heat(double T){
    return 32 * (10 * exp(16 / T) + 9 * exp(8 / T) + 5 * exp(-8 / T)) / (pow(T * (5 + exp(16 / T) + 2 * exp(-8 / T)), 2));
}
#define L 5
int main(){
    ising2D_rigor ising(L);
    ofstream fout;
    string folder = "data/";
    string name1 = "rigor_";
    string name2 = "_times_";
    string name3 = "_erglist.txt";
    string name4 = "_Cv.txt";
    string num = to_string(L);
    ising.fprint_vector_erg(folder + name1 + num + name2 + num + name3);
    fout.open(folder + name1 + num + name2 + num + name4);
    if(!fout){
        cout << "Error." << endl;
        return 0;
    }
    for (double T = 0.01; T < 20.01; T+=0.01)
    {
        //double c1 = ising.get_specific_heat(T);
        double E = ising.get_mean_erg(T);
        double Cv = ising.get_specific_heat(T);
        fout << fixed << setprecision(2) << setw(5) << T << "\t";
        fout << scientific << setprecision(6) << setw(15) << E << "\t" << setw(15) << Cv << endl;
    }
    fout.close();
    return 0;
}

