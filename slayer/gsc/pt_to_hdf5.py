import torch
from gsc_train import Network

if __name__ == '__main__':
	net = Network()

	# dummy workload to init neuron shape
	net(torch.zeros(5, 20, 201))

	net.load_state_dict(torch.load('kangaroo_gsc_trained_weight_norm/network.pt'))
	net.export_hdf5('kangaroo_gsc_trained_weight_norm/network.h5')
	print('Network exported to kangaroo_gsc_trained_weight_norm/network.h5')
	