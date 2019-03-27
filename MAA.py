import torch
from torch import nn



class MAA_layer(nn.Module):
	"""
	Ps: a list of probabilities
	Xs: a list of extracted features
	
	q[t+1, i] := p[t+1] q[t, i-1] + (1-p[t+1]) q[t, i]
	boundary condition: q[t, -1] = 0, q[t, t+1] = 0, q[0, 0] = 1
	
	m[t+1, i] = (i-1) p[t+1] m[t, i-1] / i + (1-p[t+1]) m[t, i] + p[t+1] q[t, i-1] X[t+1] / i
	boundary condition: m[t, 0] = 0, m[t, t+1] = 0

	t: the frame number
	i: the count of actions in a squence of frames
	"""
	def __init__(self) :
		super(MAA_layer, self).__init__()
		
	def forward(self, p, x):   # input: p (1 * T * 1)    x (1 * T * feat_dim)
		ps = p.squeeze(2).squeeze(0)    # p (T)
		xs = x.squeeze(0)   # x (T * feat_dim)

		T = xs.size()[0]  # the length of the sequence 

		# starting index of qs: -1
		# starting index of ms: 0
		prev_qs, prev_ms = [0.0, 1.0, 0.0], [0.0, 0.0]	
		num_prev_qs, num_prev_ms = len(prev_qs), len(prev_ms)
		qs, ms = [], []

		for t in range(T):
			qs += [0.0]
			for i in range(1, num_prev_qs):
				q = ps[t].clone() * prev_qs[i-1] + (1-ps[t].clone()) * prev_qs[i]
				qs += [q]
			qs += [0.0]

			ms += [0.0]
			for i in range(1, num_prev_ms):
				m = (i - 1.0) / i * ps[t].clone() * prev_ms[i-1] + (1 - ps[t].clone()) * prev_ms[i] + 1.0 / i * ps[t].clone() * prev_qs[i] * xs[t].clone()
				
				ms += [m]
			ms += [0.0]
			prev_qs, prev_ms = qs, ms
			num_prev_qs, num_prev_ms = len(prev_qs), len(prev_ms)
			qs, ms = [], []

		MAA_feature = []
		for i in range(1, len(prev_ms) - 1):
			MAA_feature.append(prev_ms[i].unsqueeze(0))
		MAA_feature = torch.cat(MAA_feature, dim=0)
		MAA_feature = torch.sum(MAA_feature, dim=0)
		return MAA_feature

if __name__ == '__main__' : 
	f = MAA_layer()
	p = torch.randn(1, 10, 1).cuda()
	x = torch.randn(1, 10, 64).cuda()
	print (f(p,x))

		
