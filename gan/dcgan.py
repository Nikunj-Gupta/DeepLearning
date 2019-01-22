import numpy as np 
import time 
from tensorflow.examples.tutorials.mnist import input_data 

from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Reshape 
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D 
from keras.layers import LeakyReLU, Dropout 
from keras.layers import BatchNormalization 
from keras.optimizers import Adam, RMSprop 


class DCGAN(object): 
	def __init__(self, img_rows=28, img_cols=28, channel=1): 
		self.img_rows = img_rows 
		self.img_cols = img_cols 
		self.channel = channel 
		self.D = None 
		self.G = None 
		self.AM = None 
		self.DM = None 

	def discriminator(self): 
		if self.D: 
			return self.D 
		self.D = Sequential() 
		depth = 64 
		dropout = 0.4 
		input_shape = (self.img_rows, self.img_cols, self.channel) 
		self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same')) 
		self.D.add(LeakyReLU(alpha=0.2)) 
		self.D.add(Dropout(dropout)) 

		self.D.add(Conv2D(depth*2, 5, strides=2, padding='same')) 
		self.D.add(LeakyReLU(alpha=0.2)) 
		self.D.add(Dropout(dropout)) 
		
		self.D.add(Conv2D(depth*4, 5, strides=2, padding='same')) 
		self.D.add(LeakyReLU(alpha=0.2)) 
		self.D.add(Dropout(dropout)) 
		
		self.D.add(Conv2D(depth*8, 5, strides=2, padding='same')) 
		self.D.add(LeakyReLU(alpha=0.2)) 
		self.D.add(Dropout(dropout)) 

		self.D.add(Flatten()) 
		self.D.add(Dense(1)) 
		self.D.add(Activation('sigmoid')) 
		self.D.summary() 
		#print ("Discriminator Done")
		return self.D 

	def generator(self): 
		if self.G: 
			return self.G 
		self.G = Sequential() 
		dropout = 0.4 
		depth = 4*64 
		dim = 7 

		self.G.add(Dense(dim*dim*depth, input_dim=100)) 
		self.G.add(BatchNormalization(momentum=0.9)) 
		self.G.add(Activation('relu')) 
		self.G.add(Reshape((dim, dim, depth))) 
		self.G.add(Dropout(dropout)) 

		self.G.add(UpSampling2D()) 
		self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same')) 
		self.G.add(BatchNormalization(momentum=0.9)) 
		self.G.add(Activation('relu')) 

		self.G.add(UpSampling2D()) 
		self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same')) 
		self.G.add(BatchNormalization(momentum=0.9)) 
		self.G.add(Activation('relu')) 

		self.G.add(UpSampling2D()) 
		self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same')) 
		self.G.add(BatchNormalization(momentum=0.9)) 
		self.G.add(Activation('relu')) 

		self.G.add(Conv2DTranspose(1, 5, padding='same')) 
		self.G.add(Activation('sigmoid')) 
		self.G.summary() 
		#print ("Generator Done") 
		return self.G 

	def discriminator_model(self): 
		if self.DM: 
			return self.DM 
		optimizer = RMSprop(lr=0.0002, decay=6e-8) 
		self.DM = Sequential() 
		self.DM.add(self.discriminator()) 
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
		#print("Discriminator Model Done") 
		return self.DM 

	def adversarial_model(self): 
		if self.AM: 
			return self.AM 
		optimizer = RMSprop(lr=0.0001, decay=3e-8) 
		self.AM = Sequential() 
		self.AM.add(self.generator()) 
		self.AM.add(self.discriminator()) 
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
		#print("Adversarial Model Done") 
		return self.AM 


if __name__ == '__main__': 
	dcgan = DCGAN() 
	#dcgan.discriminator() 
	#dcgan.generator() 
	#dcgan.discriminator_model() 
	#dcgan.adversarial_model() 