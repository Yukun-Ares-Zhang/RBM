import numpy as np 
#for plotting 
import matplotlib.pyplot as plt
from utils import tile_raster_images
import pyalps
import sys, subprocess, os, socket

def plot_weights(W, scale_rows_to_unit_interval=False):
    L = int(np.sqrt(W.shape[0])) # image size = L**2 
    M = int(np.sqrt(W.shape[1])) # n_features = M**2 
    #print W.shape, L, M 
    # Construct image from the weight matrix
    image= tile_raster_images(
            X=W.T,
            img_shape=(L, L),
            tile_shape=(M, M),
            tile_spacing=(1, 1), 
            output_pixel_vals = False, 
            scale_rows_to_unit_interval = scale_rows_to_unit_interval
        )

    fig = plt.figure()  
    image = np.ma.array (image, mask=np.isnan(image))
    cmap = plt.cm.RdBu_r
    cmap.set_bad('white',alpha=1.)
    im = plt.imshow(image, cmap=cmap, interpolation='none')
    plt.colorbar(im)
    plt.axis('off')

if __name__=='__main__':
    import numpy as np 
    import h5py 

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', default='/Users/wanglei/Src/RBM/data/nnfit/fk_L8_U4_T0.15_nhidden64_/weights.hdf5')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-show", action='store_true',  help="show figure right now")
    group.add_argument("-outname", default="result.pdf",  help="output pdf file")

    args = parser.parse_args()

    h5 = h5py.File(args.filename,'r')
    try:
        W = np.array(h5['mylayer_1/mylayer_1_W'][()])
        b = np.array(h5['mylayer_1/mylayer_1_b'][()])
    except KeyError:
        #W = np.array(h5['mylayer_1/mylayer_1_W'][()])
        #b = np.array(h5['mylayer_1/mylayer_1_b'][()])

        W = np.array(h5['convolution2d_1/convolution2d_1_W'][()])
        W.shape = (W.shape[0], W.shape[-1])
        W = W.transpose()
        b = np.array(h5['convolution2d_1/convolution2d_1_b'][()])

    h5.close() 

    plot_weights(W, scale_rows_to_unit_interval=True)

if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=600, transparent=True)
    pass 
    
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
