import scipy.sparse as sps 
from pltgraph import parse 
import scipy.sparse as sps 
from numpy import where, array  , zeros 
import subprocess 

def write_input( latticename, L, W):
    '''
    write input file for a lattice 
    '''

    input='''LATTICE_LIBRARY = "mylattices.xml"
LATTICE = "%s"
L = %g
W = %g 
'''%(latticename, L, W)
    
    inputfile = 'lattice.input'
    f = file(inputfile,'w')
    f.write(input)
    f.close()

    return inputfile 


class Lattice:
    def __init__(self, latticename, L, W):

        #generate lattice input file 
        inputfile = write_input(latticename, L,W)
        cmd = ['printgraph', inputfile]
        graphfile = inputfile.replace('.input','.graph')
        f = file(graphfile,'w')
        f.write(subprocess.check_output(cmd))
        f.close()
        
        # and parse it 
        graph = parse(graphfile)
 
        #read lattice 
        Row = []
        Col = []
        Dat = []
        edges = graph[1]
        for edge in edges.values():
            Row.append(int(edge['source'])-1)
            Col.append(int(edge['target'])-1)
            Dat.append(int(edge['type']))
        
        #print Row
        #print Col
        #print Dat  #stores bond type 

        self.Nsite = max(max(Row), max(Col)) + 1
        
        #Kinetic part 
        Val = zeros(len(Dat), float)
        Dat = array(Dat)
        Val[where(Dat==0)]= -1.0 
        Val[where(Dat==1)]= 1.0 
    
        self.Kmat = sps.csr_matrix((Val, (Row, Col)), shape=(self.Nsite, self.Nsite)).todense()
        self.Kmat += self.Kmat.transpose()
        self.Kmat = sps.csr_matrix(self.Kmat)
       
        #import matplotlib.pyplot as plt 
        #plt.spy(self.Kmat, markersize=10 ,marker='.')
        #plt.show()

if __name__=='__main__':

    latticename = 'square lattice'
    L = 3
    W = 2

    lattice = Lattice(latticename, L, W)
    print lattice.Kmat.todense() 
    from scipy.linalg import eigh 
    w, v = eigh(lattice.Kmat.todense())

    import numpy as np 
    index = np.where(w<0)
    print w[index].sum() 
