#ifndef ISING_H
#define ISING_H

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stack>
#include <iostream>

using namespace std;

class ising2D
{
private:
    int m_L, m_N, m_XNN, m_YNN;
    int *m_s;
    double m_prob[5];
    double m_beta, m_Padd;

public:
    ising2D(int L, int mode, double T);
    ~ising2D();
    void sweep();
    void step();
    void change_T(double new_T);
    double magnetization();
    double Hamitonian();
};
ising2D::ising2D(int L, int mode, double T) : m_L(L), m_N(L * L), m_XNN(1), m_YNN(L), m_beta(1 / T), m_Padd(1 - exp(-2 * m_beta))
{
    int i;
    for (i = 2; i < 5; i += 2)
    {
        m_prob[i] = exp(-2 * m_beta * i);
    }
    m_s = new int[m_N];
    switch (mode)
    {
    case -1: // All spin -1
        for (i = 0; i < m_N; i++)
        {
            m_s[i] = -1;
        }
        break;

    case 1: // All spin +1
        for (i = 0; i < m_N; i++)
        {
            m_s[i] = 1;
        }
        break;

    default: // configuration at infinite T
        for (i = 0; i < m_N; i++)
        {
            m_s[i] = 2 * (rand() % 2) - 1;
        }
        break;
    }
}
ising2D::~ising2D(){
    delete[] m_s;
}
void ising2D::sweep()
{
    int i, k;
    int nn, sum = 0, delta;
    for (k = 0; k < m_N; k++)
    {
        sum = 0;
        i = rand() % m_N;
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
        delta = sum * m_s[i];
        if (delta <= 0)
            m_s[i] = -m_s[i];
        else if ((rand() / (RAND_MAX + 1.0)) < m_prob[delta])
            m_s[i] = -m_s[i];
    }
}
void ising2D::step(){
    int i;
    int oldspin, newspin;
    int current, nn;
    stack<int> index;

    i = rand() % m_N;
    index.push(i);
    oldspin = m_s[i];
    newspin = - m_s[i];
    m_s[i] = newspin;
    while(!index.empty()){
        // cout << "stack size: " << index.size() << endl;
        current = index.top();
        index.pop();
        if((nn = current + m_XNN) >= m_N)
            nn -= m_N;
        if(m_s[nn] == oldspin)
            if((rand() / (RAND_MAX+1.0)) < m_Padd){
                index.push(nn);
                m_s[nn] = newspin;
            }

        if ((nn = current - m_XNN) < 0)
            nn += m_N;
        if (m_s[nn] == oldspin)
            if ((rand() / (RAND_MAX + 1.0)) < m_Padd)
            {
                index.push(nn);
                m_s[nn] = newspin;
            }

        if ((nn = current + m_YNN) >= m_N)
            nn -= m_N;
        if (m_s[nn] == oldspin)
            if ((rand() / (RAND_MAX + 1.0)) < m_Padd)
            {
                index.push(nn);
                m_s[nn] = newspin;
            }

        if ((nn = current - m_YNN) < 0)
            nn += m_N;
        if (m_s[nn] == oldspin)
            if ((rand() / (RAND_MAX + 1.0)) < m_Padd)
            {
                index.push(nn);
                m_s[nn] = newspin;
            }
    }
}
void ising2D::change_T(double new_T){
    m_beta = 1 / new_T;
    m_Padd = 1 - exp(-2 * m_beta);
}
double ising2D::magnetization()
{
    int i;
    double m = 0;
    for (i = 0; i < m_N; i++)
    {
        m += m_s[i];
    }
    m /= m_N;
    return m;
}
double ising2D::Hamitonian(){
    double hamitonian = 0;
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

#endif