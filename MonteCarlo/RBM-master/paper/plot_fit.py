import numpy as np 
import matplotlib.pyplot as plt 
from config import * 
import argparse 
import os, sys 
import subprocess 
import pyalps 
import pyalps.plot
import socket

parser = argparse.ArgumentParser()
parser.add_argument('-filename', default='/Users/wanglei/Src/RBM/data/nnfit/fk_L8_U4_T0.15_nhidden64_/fit.dat')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-show", action='store_true',  help="show figure right now")
group.add_argument("-outname", default="result.pdf",  help="output pdf file")

args = parser.parse_args()

i, ytest, ypred = np.loadtxt(args.filename, unpack=True)

#offset = ytest.min()
#ytest -= offset 
#ypred -= offset 

idx = np.argsort(ytest)[::-1]

fig = plt.figure(figsize=(8, 5))
plt.plot(ypred[idx], 'o', alpha=0.5, label='RBM') 
plt.plot(ytest[idx], 'r-', lw=2, label='Ising')
plt.legend(loc='lower left')
plt.ylabel(r'$-F(\mathbf{x})$')
plt.xlabel('Test sample')
#plt.ylim([0, 50])
#plt.xlim([0, 10200])
plt.subplots_adjust(bottom=0.15)

if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=300, transparent=True)
    
    #email it to me 
    recipient = "lewang@phys.ethz.ch"
    message = 'Send from ' + os.getcwd() + ' with python ' + ' '.join([str(a) for a in sys.argv])
    subject = 'Figure: ' + args.outname

    machinename = socket.gethostname()
    if 'brutus' in machinename or 'monch' in machinename:
        pyalps.sendmail(recipient    # email address of recipients 
                       , subject = subject 
                       , message = message 
                       , attachment= args.outname 
                       )
    else:
        cmd = ['sendmail.py', '-t', recipient+',', '-s', 'Automatic email message from ALPS. '+ subject , '-m', message, '-a', args.outname]
        subprocess.check_call(cmd)

    os.system('rm '+args.outname)
