import argparse
import re 
from load_data import load_data

#for plotting 
try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import tile_raster_images

parser = argparse.ArgumentParser(description='')
parser.add_argument("-filename", default="ising_L28_T2.3.dat", help="filename")
args = parser.parse_args()

X_train, y_train, X_test, y_test = load_data(args.filename) 


L = int(re.search('_L([0-9]*)_',args.filename).group(1)) 
T = float(re.search('T(-?[0-9]*\.?[0-9]*).dat',args.filename).group(1)) 

M = 8 
# Construct image from the weight matrix
image = Image.fromarray(
    tile_raster_images(
        X=X_train[:M*M],
        img_shape=(L, L),
        tile_shape=(M, M),
        tile_spacing=(1, 1)
    )
)
image.save('configs_T'+str(T)+'.png')
