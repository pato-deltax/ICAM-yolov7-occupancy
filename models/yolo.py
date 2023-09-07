import torch
import torch.nn as nn


class IKeypoint(nn.Module):
	# stride = None  # strides computed during build
	export = False  # onnx export

	def __init__(self):  # detection layer
		super(IKeypoint, self).__init__()
		self.m = nn.ModuleList(nn.Conv2d())  # output conv

	def forward(self, x):
		z = []  # inference output
		self.training |= self.export
		for i in range(self.nl):
			if self.nkpt is None or self.nkpt==0:
				x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
			else :
				x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

			bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
			x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
			x_det = x[i][..., :6]
			x_kpt = x[i][..., 6:]

			if not self.training:  # inference
				if self.grid[i].shape[2:4] != x[i].shape[2:4]:
					self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
				kpt_grid_x = self.grid[i][..., 0:1]
				kpt_grid_y = self.grid[i][..., 1:2]

				if self.nkpt == 0:
					y = x[i].sigmoid()
				else:
					y = x_det.sigmoid()

				if self.inplace:
					xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
					wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
					if self.nkpt != 0:
						x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
						x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
						x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

					y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

				else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
					xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
					wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
					if self.nkpt != 0:
						y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1,1,1,1,self.nkpt))) * self.stride[i]  # xy
					y = torch.cat((xy, wh, y[..., 4:]), -1)

				z.append(y.view(bs, -1, self.no))
		return x if self.training else (torch.cat(z, 1), x)

	@staticmethod
	def _make_grid(nx=20, ny=20):
		yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
		return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

	def forward(self, x):
		return self.forward_once(x)  # single-scale inference, train

	def forward_once(self, x):
		y = []  # outputs
		for m in self.model:
			if m.f != -1:  # if not from previous layer
				x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
			x = m(x)  # run
			y.append(x if m.i in self.save else None)  # save output
		return x
