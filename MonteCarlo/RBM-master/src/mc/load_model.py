import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from load_data import load_data
import plot 

img_size = 28 
original_dim=784
intermediate_dim=512
latent_dim = 512

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

generator.summary() 
generator.load_weights('generator.h5')

z_sample = np.random.randn(1, latent_dim)
x_decoded = generator.predict(z_sample)
digit = x_decoded[0].reshape(img_size, img_size)

print ( digit )
