import torch
from gsc_train import Network

if __name__ == '__main__':
	net = Network()
	net.load_state_dict(torch.load('gsc_trained/network.pt'))
	net.export_hdf5('gsc_trained/network.h5')
	print('Network exported to gsc_trained/network.h5')