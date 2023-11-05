import numpy as np 
#for plotting 
import matplotlib.pyplot as plt
from utils import tile_raster_images
from matplotlib.ticker import MaxNLocator

def plot_weights(W, epoch=0, resfolder='./'):
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
            scale_rows_to_unit_interval =False
        )

    fig = plt.figure()  
    image = np.ma.array (image, mask=np.isnan(image))
    cmap = plt.cm.RdBu_r
    cmap.set_bad('white',alpha=1.)
    im = plt.imshow(image, cmap=cmap, interpolation='none')
    plt.colorbar(im)
    plt.axis('off')

    plt.savefig(resfolder+'/filters_at_epoch_%i.png' % epoch)

if __name__=='__main__':
    import numpy as np 
    W = np.random.randn(256, 16)

    plot_weights(W)

