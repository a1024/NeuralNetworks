import ctypes
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import time
from datetime import timedelta
import os
import torchvision
from PIL import Image
import numpy


from ctypes import cdll
from sys import platform

if platform == "linux" or platform == "linux2":
	libac=cdll.LoadLibrary('./libac.so')#arithmetic coder
elif platform == "win32":
	libac=cdll.LoadLibrary('./libac.dll')

class Buffer(ctypes.Structure):
	_fields_=[
		('size', ctypes.c_longlong),
		('cap', ctypes.c_longlong),
		('data', ctypes.POINTER(ctypes.c_char)),
	]

c_int_p=ctypes.POINTER(ctypes.c_int)

libac.encode_bytes.argtypes=[
	ctypes.c_char_p,
	ctypes.c_longlong,
	ctypes.POINTER(Buffer),
	ctypes.c_int
]

libac.encode_bytes.restype=ctypes.c_int
libac.decode_bytes.argtypes=[
	ctypes.c_char_p,
	ctypes.c_longlong,
	ctypes.c_longlong,
	ctypes.c_char_p,
	ctypes.c_int,
	ctypes.c_int
]
libac.decode_bytes.restype=ctypes.c_int

libac.free_memory.argtypes=[ctypes.c_void_p]

libac.test.argtypes=[
	ctypes.POINTER(ctypes.c_int32),
	ctypes.c_longlong,
	ctypes.c_int,
]
libac.test.restype=ctypes.c_int


def save_model(model, iter, name):#https://github.com/liujiaheng/compression
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_model(model, f):#https://github.com/liujiaheng/compression
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class CompressorModel(nn.Module):
	def __init__(self):
		super(CompressorModel, self).__init__()
		self.activ=nn.LeakyReLU()
		
		# v2
		self.analysis_conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=6, stride=2, padding=2, groups=1)
		self.analysis_conv2=nn.Conv2d(in_channels=12, out_channels=40, kernel_size=4, stride=2, padding=2, groups=4)

		self.analysis_conv_mean=nn.Conv2d(in_channels=40, out_channels=128, kernel_size=3, stride=2, padding=1, groups=8)
		self.analysis_conv_std=nn.Conv2d(in_channels=40, out_channels=128, kernel_size=3, stride=2, padding=1, groups=8)

		self.synth_conv1=nn.ConvTranspose2d(in_channels=128, out_channels=40, kernel_size=3, stride=2, padding=1, groups=8)
		self.synth_conv2=nn.ConvTranspose2d(in_channels=40, out_channels=12, kernel_size=4, stride=2, padding=2, groups=4)
		self.synth_conv3=nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=6, stride=2, padding=2, groups=1)


		# v1
		#self.analysis_conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1, groups=1)
		#self.analysis_conv2=nn.Conv2d(in_channels=12, out_channels=48, kernel_size=4, stride=2, padding=1, groups=4)
		#self.analysis_conv3=nn.Conv2d(in_channels=48, out_channels=192, kernel_size=4, stride=2, padding=1, groups=16)
		#self.analysis_conv4=nn.Conv2d(in_channels=192, out_channels=768, kernel_size=4, stride=2, padding=1, groups=64)
		#
		#self.analysis_conv_mean=nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1, groups=256)
		#self.analysis_conv_std=nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1, groups=256)
		#
		#self.synth_conv1=nn.ConvTranspose2d(in_channels=768, out_channels=192, kernel_size=4, stride=2, padding=1, groups=64)
		#self.synth_conv2=nn.ConvTranspose2d(in_channels=192, out_channels=48, kernel_size=4, stride=2, padding=1, groups=16)
		#self.synth_conv3=nn.ConvTranspose2d(in_channels=48, out_channels=12, kernel_size=4, stride=2, padding=1, groups=4)
		#self.synth_conv4=nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1, groups=1)


		self.U=torch.distributions.Uniform(-0.5, 0.5)
		self.N=torch.distributions.Normal(0, 1)
		self.bitratemap=0
		self.ReLU=nn.ReLU()
		self.bitrate=0

		self.entropy=0.

	def encode_start(self, x, testtime):
		x=self.analysis_conv1(x)
		x=self.activ(x)
		x=self.analysis_conv2(x)
		x=self.activ(x)
		#x=self.analysis_conv3(x)
		#x=self.activ(x)
		#x=self.analysis_conv4(x)
		#x=self.activ(x)

		if testtime==0:
			x+=self.U.sample(x.shape)		# quantization noise

		mean=self.analysis_conv_mean(x)	# VAE
		self.bitratemap=self.ReLU(self.analysis_conv_std(x))+1e-7	#log2(stddev) ~= bitrate > 0
		self.bitrate=torch.mean(self.bitratemap)	# bit rate is proportional to stddev

		std=torch.pow(2, self.bitratemap)

		return mean, std

	def decode_end(self, x):
		x=self.synth_conv1(x)
		x=self.activ(x)
		x=self.synth_conv2(x)
		x=self.activ(x)
		x=self.synth_conv3(x)
		#x=self.activ(x)
		#x=self.synth_conv4(x)

		x=torch.clamp(x, 0, 1)
		return x

	def forward(self, x):
		mean, std=self.encode_start(x, 0)

		x = mean + std*torch.normal(torch.zeros(mean.shape), 1)

		x=self.decode_end(x)
		return x

	def test(self, x):
		with torch.no_grad():
			mean, std=self.encode_start(x, 1)

			compr=torch.round(mean)

			x=self.decode_end(compr)
		return x, compr

def save_tensor_as_grid(x, nrows, name):
	grid=torchvision.utils.make_grid(x*255, nrow=nrows)
	grid=grid.permute(1, 2, 0)
	grid=grid.numpy()
	grid=grid.astype('uint8')
	image=Image.fromarray(grid)
	image.save(name, format='PNG')


pretrained=1
epochs=0
lr=0.0001
output_size=50

test_size=output_size*output_size
train_size=50000-test_size
batch_size=train_size//200


model=CompressorModel()
if pretrained:
	load_model(model, 'iter_0.pth.tar')

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model parameters: %d'%pytorch_total_params)

optimizer=optim.Adam(params=model.parameters(), lr=lr)
loss=nn.MSELoss()

train_data=datasets.CIFAR10('../data', train=True, download=True, transform=transforms.ToTensor())
train, test=random_split(train_data, [train_size, test_size])
train_loader=DataLoader(train, batch_size=batch_size)
test_loader=DataLoader(test, batch_size=test_size)

start=time.time()

for epoch in range(epochs):
	progress=0
	mse=0
	bitrate=0
	for x, y in train_loader:#train
		xhat=model(x)		#1 forward

		J=loss(xhat, x)		#2 compute the objective function
		current_mse=J.item()
		#J+=model.bitrate

		mse+=current_mse
		bitrate+=model.bitrate

		model.zero_grad()	#3 cleaning the gradients

		J.backward()		#4 accumulate the partial derivatives of J wrt params
		optimizer.step()	#5 step in the opposite direction of the gradient

		progress+=batch_size
		print('%d/%d = %.2f%%, mse=%f rate=%f, loss=%f\t\t'%(progress, train_size, 100*progress/train_size, current_mse, model.bitrate, current_mse+model.bitrate), end='\r')

	save_model(model, iter=0, name='')

	t2=time.time()
	print('\t\t\t\t', end='\r')
	niter=progress/batch_size
	mse/=niter
	bitrate/=niter
	#print('Epoch %d loss=mse %f rate %f'%(epoch+1, mse, bitrate), end=' ')
	print('Epoch %d mse %f rate %f, loss %f'%(epoch+1, mse, bitrate, mse+bitrate), end=' ')
	print('elapsed %f'%(t2-start), end=' ')
	print(str(timedelta(seconds=t2-start)))

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

test_loss=0
for x, y in test_loader:
	xhat, code=model.test(x)

	code=code.detach()
	code=code.numpy()
	usize=code.size
	code=code.astype(numpy.int32)
	#print(code)#
	code=code.ctypes.data_as(c_int_p)
	csize=libac.test(code, usize, 1)
	print('Compression %d->%d bytes = %f'%(usize, csize, usize/csize))

	save_tensor_as_grid(x, output_size, 'compression_original.png')
	save_tensor_as_grid(xhat, output_size, 'compression_retrieved.png')

	J=loss(xhat, x)
	test_loss+=J.item()
	
#test_loss/=test_size
print('Test loss: %f'%test_loss)
