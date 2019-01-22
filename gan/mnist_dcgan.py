import numpy as np 
import time 
from tensorflow.examples.tutorials.mnist import input_data 

from dcgan import DCGAN 


class MNIST_DCGAN(object): 
	def __init__(self): 
		self.img_rows = 28 
		self.img_cols = 28 
		self.channel = 8 

		self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images 
		self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32) 

		self.DCGAN = DCGAN() 
		self.discriminator = self.DCGAN.discriminator_model() 
		self.adversarial = self.DCGAN.adversarial_model() 
		self.generator = self.DCGAN.generator() 




if __name__ == '__main__': 
	mnist_dcgan = MNIST_DCGAN() 