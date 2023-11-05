#include <alps/alea.h>
#include <alps/parameter.h>
#include <alps/lattice.h>
#include <boost/foreach.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include <cmath>
#include <iostream>
#include <stack>
#include <vector>

///compute total energy 
double calc_energy(const alps::graph_helper<>& lattice, const std::vector<int>& spins){
    double res = 0; 
    for (unsigned si=0; si<lattice.num_sites(); ++si){
        BOOST_FOREACH(alps::graph_helper<>::site_descriptor const& sj, lattice.neighbors(si)) {
            res -= spins[sj] * spins[si]; 
        }
    }
    return res/2.;  
}

int main() {

  alps::Parameters params(std::cin);
  const int MCSTEP = params.value_or_default("SWEEPS", 1 << 15);
  const int MCTHRM = params.value_or_default("THERMALIZATION", MCSTEP >> 3);
  const int NSKIP = params.value_or_default("NSKIP", 100);
  const unsigned int SEED = params.value_or_default("SEED", 93812);
  const double T = params.value_or_default("T", 0.1);

  //const double Tc = 2.269; 

  // setting up square lattice
  alps::graph_helper<> graph(params);
  const int N = graph.num_sites();

  // random number generator
  boost::mt19937 eng(SEED);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
    random_01(eng, boost::uniform_real<>());

  // spin configuration
  std::vector<int> spin(N, 1);
  //double E0 = calc_energy(graph, spin); //lowest energy  
  //std::cout << "E0 " <<  E0 << std::endl; 
  // stack for uninspected sites
  std::stack<int> stck;

    // connecting probability
    double pc = 1 - std::exp(-2./T);

    for (int mcs = 0; mcs < MCSTEP + MCTHRM; ++mcs) {
      int s = static_cast<int>(random_01() * N);
      int so = spin[s];
      spin[s] = -so;
      stck.push(s);
      int cs = 0;
      while (!stck.empty()) {
        ++cs;
        int sc = stck.top();
        stck.pop();
        BOOST_FOREACH(alps::graph_helper<>::site_descriptor const& sn, graph.neighbors(sc)) {
          if (spin[sn] == so && random_01() < pc) {
            stck.push(sn);
            spin[sn] = -so;
          }
        }
      }
      
      //output spin configurations 
      if (mcs >= MCTHRM && mcs%NSKIP==0){

         //output spins
         for (unsigned si=0; si< spin.size(); ++si){
             std::cout << spin[si] << " ";
         }

         std::cout << -calc_energy(graph, spin)/T << std::endl; 

         //output spin product    
         //alps::graph_helper<>::bond_iterator it, it_end;
         //for (boost::tie(it, it_end)=graph.bonds(); it!=it_end; ++it){
         //    alps::graph_helper<>::site_descriptor si = source(*it,graph.graph()); 
         //    alps::graph_helper<>::site_descriptor sj = target(*it,graph.graph()); 
         //    std::cout << spin[si] * spin[sj] << " "; 
         //}
         //std::cout << T << std::endl;
         
         //unsigned label = (T< Tc) ? 0: 1; 
         //std::cout << label << std::endl; 
      }

    }// loop over MC sweeps 
  return 0;
}
