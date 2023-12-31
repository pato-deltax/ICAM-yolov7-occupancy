import torch
import torch.nn as nn

##### basic #####

def autopad(k, p=None):  # kernel, padding
	# Pad to 'same'
	if p is None:
		p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
	return p


class ReOrg(nn.Module):
	def __init__(self):
		super(ReOrg, self).__init__()

	def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
		return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
	def __init__(self, dimension=1):
		super(Concat, self).__init__()
		self.d = dimension

	def forward(self, x):
		return torch.cat(x, self.d)


class Conv(nn.Module):
	# Standard convolution
	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
		super(Conv, self).__init__()
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
		self.bn = nn.BatchNorm2d(c2)
		self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

	def forward(self, x):
		return self.act(self.bn(self.conv(x)))

	def fuseforward(self, x):
		return self.act(self.conv(x))
	
##### end of basic #####
##### cspnet #####

class SPPCSPC(nn.Module):
	# CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
	def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
		super(SPPCSPC, self).__init__()
		c_ = int(2 * c2 * e)  # hidden channels
		self.cv1 = Conv(c1, c_, 1, 1)
		self.cv2 = Conv(c1, c_, 1, 1)
		self.cv3 = Conv(c_, c_, 3, 1)
		self.cv4 = Conv(c_, c_, 1, 1)
		self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
		self.cv5 = Conv(4 * c_, c_, 1, 1)
		self.cv6 = Conv(c_, c_, 3, 1)
		self.cv7 = Conv(2 * c_, c2, 1, 1)

	def forward(self, x):
		x1 = self.cv4(self.cv3(self.cv1(x)))
		y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
		y2 = self.cv2(x)
		return self.cv7(torch.cat((y1, y2), dim=1))

##### end of cspnet #####
##### yolor #####

class ImplicitA(nn.Module):
	def __init__(self, channel, mean=0., std=.02):
		super(ImplicitA, self).__init__()
		self.channel = channel
		self.mean = mean
		self.std = std
		self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
		nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

	def forward(self, x):
		return self.implicit + x
	

class ImplicitM(nn.Module):
	def __init__(self, channel, mean=0., std=.02):
		super(ImplicitM, self).__init__()
		self.channel = channel
		self.mean = mean
		self.std = std
		self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
		nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

	def forward(self, x):
		return self.implicit * x
	
##### end of yolor #####
