import cv2
import time
import torch
import torchvision
import numpy as np


""" ============== Global Variable Initialization ============== """

SD = (854, 480)
HD = (1280, 768)
FHD = (1920, 1080)
UHD = (3840, 2160)

ORDER_NUMBER = {
	"driver": 3,
	"front": 4,
	"right": 5,
	"center": 6,
	"left": 7,
}

""" ============== GENERAL FUNCTIONS ============== """

def xyxy2xywh(x):
	""" Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right """
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	y[:, 0] = (x[:, 0] + x[:, 2]) / 2	# x center
	y[:, 1] = (x[:, 1] + x[:, 3]) / 2	# y center
	y[:, 2] = x[:, 2] - x[:, 0]			# width
	y[:, 3] = x[:, 3] - x[:, 1]			# height
	return y


def xywh2xyxy(x):
	""" Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right """
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	y[:, 0] = x[:, 0] - x[:, 2] / 2		# top left x
	y[:, 1] = x[:, 1] - x[:, 3] / 2		# top left y
	y[:, 2] = x[:, 0] + x[:, 2] / 2		# bottom right x
	y[:, 3] = x[:, 1] + x[:, 3] / 2		# bottom right y
	return y


def box_iou(box1, box2):
	"""
	Return intersection-over-union (Jaccard index) of boxes.
	Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
	Arguments:
		box1 (Tensor[N, 4])
		box2 (Tensor[M, 4])
	Returns:
		iou (Tensor[N, M]): the NxM matrix containing the pairwise
			IoU values for every element in boxes1 and boxes2
	https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
	"""

	def box_area(box):
		# box = 4xn
		return (box[2] - box[0]) * (box[3] - box[1])

	area1 = box_area(box1.T)
	area2 = box_area(box2.T)

	# inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
	inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
	return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
						labels=(), kpt_label=False, nc=None, nkpt=None):
	""" Runs Non-Maximum Suppression (NMS) on inference results
	Returns:
		 list of detections, on (n,6) tensor per image [xyxy, conf, cls]
	"""
	if nc is None:
		nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
	xc = prediction[..., 4] > conf_thres  # candidates

	# Settings
	min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
	max_det = 300  # maximum number of detections per image
	max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
	time_limit = 10.0  # seconds to quit after
	redundant = True  # require redundant detections
	multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
	merge = False  # use merge-NMS

	t = time.time()
	output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
	for xi, x in enumerate(prediction):  # image index, image inference
		# Apply constraints
		# x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
		x = x[xc[xi]]  # confidence

		# Cat apriori labels if autolabelling
		if labels and len(labels[xi]):
			l = labels[xi]
			v = torch.zeros((len(l), nc + 5), device=x.device)
			v[:, :4] = l[:, 1:5]  # box
			v[:, 4] = 1.0  # conf
			v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
			x = torch.cat((x, v), 0)

		# If none remain process next image
		if not x.shape[0]:
			continue

		# Compute conf
		x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

		# Box (center x, center y, width, height) to (x1, y1, x2, y2)
		box = xywh2xyxy(x[:, :4])

		# Detections matrix nx6 (xyxy, conf, cls)
		if multi_label:
			i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
			x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
		else:  # best class only
			if not kpt_label:
				conf, j = x[:, 5:].max(1, keepdim=True)
				x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
			else:
				kpts = x[:, 6:]
				conf, j = x[:, 5:6].max(1, keepdim=True)
				x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


		# Filter by class
		if classes is not None:
			x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

		# Apply finite constraint
		# if not torch.isfinite(x).all():
		#     x = x[torch.isfinite(x).all(1)]

		# Check shape
		n = x.shape[0]  # number of boxes
		if not n:  # no boxes
			continue
		elif n > max_nms:  # excess boxes
			x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

		# Batched NMS
		c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
		boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
		i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
		if i.shape[0] > max_det:  # limit detections
			i = i[:max_det]
		if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
			# update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)

			iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
			weights = iou * scores[None]  # box weights
			x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
			if redundant:
				i = i[iou.sum(1) > 1]  # require redundancy

		output[xi] = x[i]
		if (time.time() - t) > time_limit:
			print(f'WARNING: NMS time limit {time_limit}s exceeded')
			break  # time limit exceeded
	return output


""" ============== DATASETS FUNCTIONS ============== """

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	""" Resize and pad image while meeting stride-multiple constraints """
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)


def output_to_keypoint(output):
	""" Convert model output to target format [batch_id, class_id, x, y, w, h, conf] """
	targets = []
	for i, o in enumerate(output):
		kpts = o[:,6:]
		o = o[:,:6]
		for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
			targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
	return np.array(targets)


""" ============== PLOTS FUNCTIONS ============== """

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
	""" Plot the skeleton and keypointsfor coco datatset """
	palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
						[230, 230, 0], [255, 153, 255], [153, 204, 255],
						[255, 102, 255], [255, 51, 255], [102, 178, 255],
						[51, 153, 255], [255, 153, 153], [255, 102, 102],
						[255, 51, 51], [153, 255, 153], [102, 255, 102],
						[51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
						[255, 255, 255], [0,0,0]])

	skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
				[7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
				[1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

	# pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
	# pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

	#Change the skeleton color to black
	pose_limb_color = palette[[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
	pose_kpt_color = palette[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	radius = 5
	num_kpts = len(kpts) // steps

	for kid in range(num_kpts):
		r, g, b = pose_kpt_color[kid]
		x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
		if not (x_coord % 640 == 0 or y_coord % 640 == 0):
			if steps == 3:
				conf = kpts[steps * kid + 2]
				if conf < 0.5:
					continue
			cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

	for sk_id, sk in enumerate(skeleton):
		r, g, b = pose_limb_color[sk_id]
		pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
		pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
		if steps == 3:
			conf1 = kpts[(sk[0]-1)*steps+2]
			conf2 = kpts[(sk[1]-1)*steps+2]
			if conf1<0.5 or conf2<0.5:
				continue
		if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
			continue
		if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
			continue
		cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


""" ============== DELTAX FUNCTIONS ============== """

def adjust_affined_frame(frame, shift=(0, 0), angle=0, scale=1):
	""" Rotate, scale and shift the frame """
	tx, ty = shift
	center = (int(HD[0]/2), int(HD[1]/2))

	# Step 1. Get the rotation matrix and then rotate and scale the frame
	M = cv2.getRotationMatrix2D(center, angle, scale)

	# Step 2. Shift the frame by horizontal and vertical
	M[0, 2] += (tx)
	M[1, 2] += (ty)

	# Step 3. Apply the affine transformation
	affined_frame = cv2.warpAffine(frame, M, HD, flags=cv2.INTER_CUBIC)
	return affined_frame


def draw_check_points_RGB(frame):
	""" Draw check point and lines to set the camera position """
	color = [(255, 255, 0), (255, 255, 0)]

	# Draw cental lines
	cv2.line(frame, (int(HD[0]/2), 0), (int(HD[0]/2), HD[1]), color[0], 1)
	cv2.line(frame, (0, int(HD[1]/2)), (HD[0], int(HD[1]/2)), color[0], 1)

	# Back cylinder "camera"
	cv2.rectangle(frame, (649, 76), (663, 83), color[0], 1, cv2.LINE_AA)

	# Draw check_points
	cv2.circle(frame, (968, 176), 3, color[1], -1)	# Window DR
	cv2.circle(frame, (336, 156), 3, color[1], -1)	# Window FP
	cv2.circle(frame, (674, 691), 3, color[1], -1)	# button "P"
	cv2.circle(frame, (673, 727), 3, color[1], -1)	# button "Q"

	# Steering wheel
	cv2.circle(frame, (1022, 685), 3, color[1], -1)	# steering wheel-top
	cv2.circle(frame, (1056, 586), 3, color[1], -1)	# steering wheel-left
	cv2.circle(frame, (823, 689), 3, color[1], -1)	# steering wheel-right
	cv2.circle(frame, (884, 612), 3, color[1], -1)	# steering wheel-bot
	cv2.circle(frame, (952, 618), 3, color[1], -1)	# steering wheel-center

	# Gearbox
	cv2.line(frame, (638, 370), (662, 370), color[0], 1)	# gearbox-top
	cv2.line(frame, (685, 397), (699, 519), color[0], 1)	# gearbox-DR
	cv2.line(frame, (604, 479), (620, 387), color[0], 1)	# gearbox-FP

	cv2.line(frame, (612, 488), (680, 514), color[0], 1)	# gearbox-bot-top
	cv2.line(frame, (698, 532), (699, 760), color[0], 1)	# gearbox-bot-DR
	cv2.line(frame, (603, 494), (547, 682), color[0], 1)	# gearbox-bot-FP

	# Seats
	cv2.line(frame, (792, 233), (851, 247), color[0], 1)	# seat driver top
	cv2.line(frame, (753, 496), (818, 493), color[0], 1)	# seat driver mid
	cv2.line(frame, (747, 709), (791, 707), color[0], 1)	# seat driver bot

	cv2.line(frame, (451, 259), (513, 251), color[0], 1)	# seat passenger top
	cv2.line(frame, (478, 497), (537, 504), color[0], 1)	# seat passenger mid
	cv2.line(frame, (433, 680), (493, 702), color[0], 1)	# seat passenger bot


def draw_check_points_IR(frame):
	""" Draw check point and lines to set the camera position """
	color = [(255, 255, 0), (255, 255, 0)]

	# Draw cental lines
	cv2.line(frame, (int(HD[0]/2), 0), (int(HD[0]/2), HD[1]), color[0], 1)
	cv2.line(frame, (0, int(HD[1]/2)), (HD[0], int(HD[1]/2)), color[0], 1)

	# Back cylinder "camera"
	cv2.rectangle(frame, (609, 82), (621, 90), color[0], 1, cv2.LINE_AA)

	# Draw check_points
	cv2.circle(frame, (932, 155), 3, color[1], -1)	# Window DR
	cv2.circle(frame, (296, 181), 3, color[1], -1)	# Window FP
	cv2.circle(frame, (649, 696), 3, color[1], -1)	# button "P"
	cv2.circle(frame, (651, 725), 3, color[1], -1)	# button "Q"

	# Gearbox
	cv2.line(frame, (598, 375), (630, 375), color[0], 1)	# gearbox-top
	cv2.line(frame, (647, 397), (663, 523), color[0], 1)	# gearbox-DR
	cv2.line(frame, (579, 401), (567, 484), color[0], 1)	# gearbox-FP

	cv2.line(frame, (570, 492), (653, 520), color[0], 1)	# gearbox-bot-top
	cv2.line(frame, (662, 530), (681, 760), color[0], 1)	# gearbox-bot-DR
	cv2.line(frame, (567, 497), (525, 656), color[0], 1)	# gearbox-bot-FP


def print_bg_text(frame, text, order=0, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1):
	""" Print white text result with black background """
	x_offset = 15
	y_offset = 25
	exra_shift = 4
	color = ((0, 0, 0), (255, 255, 255))

	line_space = y_offset * order
	org = (x_offset, y_offset+line_space)

	text_width, text_height = cv2.getTextSize(text=text, fontFace=fontFace, fontScale=fontScale, thickness=1)[0]
	start_point, end_point = (
		(org[0] - exra_shift, org[1] + exra_shift),
		(org[0] + exra_shift + text_width, org[1] - exra_shift - text_height))
	cv2.rectangle(frame, start_point, end_point, color[0], cv2.FILLED)
	cv2.putText(frame, text, org=org, fontFace=fontFace, fontScale=fontScale, color=color[1], thickness=thickness)


def show_height_weight_result(frame, key, value):
	""" Display height, weight, and upper body results """
	order = ORDER_NUMBER[key]
	print_bg_text(frame=frame, text=f"{key}: {value}", order=order)


def update_temp_info(result, last_storage, occupancy_flags):
	""" Update temp info dict with latest results """
	passengers = {"driver": 0, "front": 1, "right": 2, "middle": 3, "left": 4}
	for key, value in result.items():
		if not occupancy_flags[passengers[key]]:
			last_storage[key] = None
		elif value:
			last_storage[key] = value
	return last_storage
