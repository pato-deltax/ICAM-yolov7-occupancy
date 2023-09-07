import cv2
import torch
import numpy as np

from torchvision import transforms
from models.utils import non_max_suppression_kpt, output_to_keypoint, letterbox, plot_skeleton_kpts


HD = (1280, 768)

POSE_LANDMARKS_17_ID = {
	"NOSE": (7, 9),
	"LEFT_EYE": (10, 12),
	"RIGHT_EYE": (13, 15),
	"LEFT_EAR": (16, 18),
	"RIGHT_EAR": (19, 21),
	"LEFT_SHOULDER": (22, 24),
	"RIGHT_SHOULDER": (25, 27),
	"LEFT_ELBOW": (28, 30),
	"RIGHT_ELBOW": (31, 33),
	"LEFT_HAND": (34, 36),
	"RIGHT_HAND": (37, 39),
	"LEFT_BELT": (40, 42),
	"RIGHT_BELT": (43, 45),
	"LEFT_KNEE": (46, 48),
	"RIGHT_KNEE": (49, 51),
	"LEFT_FOOT": (52, 54),
	"RIGHT_FOOT": (55, 57),
}

# Dictionaries with temp info height results
last_height_storage = {}

def euclidean_distance(x1,y1, x2,y2):
	return ((x2-x1)**2 + (y2-y1)**2)**0.5

# Load YOLO-v7-pose model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using GPU:', torch.cuda.is_available())

weigths = torch.load('weights/yolov7-pose.pt')
yolo_pose = weigths['model'].to(dtype=torch.float32, device=device).eval()


def detect_pose_landmarks(frame):
	""" Detect pose landmarks by yolo_v7_pose"""
	# Do some transfromation
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = letterbox(frame, (HD[0]), stride=64, auto=True)[0]

	frame = transforms.ToTensor()(frame)
	frame = torch.tensor(np.array([frame.numpy()]))
	frame = frame.to(dtype=torch.float32, device=device)

	# Main process
	with torch.no_grad():
		output, _ = yolo_pose(frame)

	# From torch To list ???
	output = non_max_suppression_kpt(output, 0.25, 0.65, nc=yolo_pose.yaml['nc'],
									 nkpt=yolo_pose.yaml['nkpt'], kpt_label=True)

	# Get landmarks coordinates with their confidence
	bbox = output_to_keypoint(output)

	# Restore frame
	frame = frame[0].permute(1, 2, 0) * 255
	frame = frame.cpu().numpy().astype(np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	return frame, bbox



def recognize_seat_owner(frame, box, front_shift=100, back_shift=40, IR=-30):
	""" Recognize a seat owner of the passengers """
	# Passenger seats ROI, where RGB=0, IR=-30 because of camera position
	driver = (830+IR, 575)
	front_passenger = (455+IR, 565)
	back_middle_passenger = (650+IR, 275)
	back_left_passenger = (740+IR, 275)
	back_right_passenger = (560+IR, 275)

	# Get a current seat position depending on the belt landmarks of a current box coordinates
	LEFT_BELT = (box[POSE_LANDMARKS_17_ID["LEFT_BELT"][0]],
				 box[POSE_LANDMARKS_17_ID["LEFT_BELT"][0] + 1])
	RIGHT_BELT = (box[POSE_LANDMARKS_17_ID["RIGHT_BELT"][0]],
				  box[POSE_LANDMARKS_17_ID["RIGHT_BELT"][0] + 1])
	seat = (int((LEFT_BELT[0] + RIGHT_BELT[0]) / 2), int((LEFT_BELT[1] + RIGHT_BELT[1]) / 2))

	# Check if the current seat is located in one of the default seat positions with some shift
	driver_w = (seat[0] - front_shift < driver[0] < seat[0] + front_shift)
	driver_h = (seat[1] - front_shift < driver[1] < seat[1] + front_shift)
	front_passenger_w = (seat[0] - front_shift < front_passenger[0] < seat[0] + front_shift)
	front_passenger_h = (seat[1] - front_shift < front_passenger[1] < seat[1] + front_shift)
	back_middle_passenger_w = (seat[0] - back_shift < back_middle_passenger[0] < seat[0] + back_shift)
	back_middle_passenger_h = (seat[1] - back_shift < back_middle_passenger[1] < seat[1] + back_shift)
	back_left_passenger_w = (seat[0] - back_shift < back_left_passenger[0] < seat[0] + back_shift)
	back_left_passenger_h = (seat[1] - back_shift < back_left_passenger[1] < seat[1] + back_shift)
	back_right_passenger_w = (seat[0] - back_shift < back_right_passenger[0] < seat[0] + back_shift)
	back_right_passenger_h = (seat[1] - back_shift < back_right_passenger[1] < seat[1] + back_shift)

	# Show default seat positions and the current seat position ("C")
	cv2.putText(frame, "C", seat, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
	cv2.putText(frame, "D", driver, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
	cv2.putText(frame, "F", front_passenger, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
	cv2.putText(frame, "M", back_middle_passenger, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
	cv2.putText(frame, "L", back_left_passenger, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
	cv2.putText(frame, "R", back_right_passenger, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)

	# Show the default seat_owner area
	passengers = [driver, front_passenger, back_middle_passenger, back_left_passenger, back_right_passenger]
	for i, center in enumerate(passengers):
		shift = front_shift if i in [0, 1] else back_shift
		start_point = (center[0] - shift, center[1] - shift)
		end_point = (center[0] + shift, center[1] + shift)
		cv2.rectangle(frame, start_point, end_point, (255, 255, 255), 1)

	# Define the current seat owner
	if driver_w and driver_h:
		seat_owner = (0, "Driver")
	elif front_passenger_w and front_passenger_h:
		seat_owner = (1, "Front passenger")
	elif back_middle_passenger_w and back_middle_passenger_h:
		seat_owner = (2, "Back middle passenger")
	elif back_left_passenger_w and back_left_passenger_h:
		seat_owner = (3, "Back left passenger")
	elif back_right_passenger_w and back_right_passenger_h:
		seat_owner = (4, "Back right passenger")
	else:
		seat_owner = (None, "Not passenger")
	return seat_owner

if __name__ == '__main__':
	cap = cv2.VideoCapture("samples/test.mp4")
	#cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	while (cap.isOpened):
		success, frame = cap.read()
		if not success:
			break

		# Get pose landmarks
		frame, bbox = detect_pose_landmarks(frame)

		occupancy_flags = [False, False, False, False, False]
		# =======================================================================================================
		for box in bbox:
			plot_skeleton_kpts(frame, box[7:].T, 3)
			points = box[7:].T
			points = points.reshape((17,3))

			eye = points[2][:2]
			eye = [int(w) for w in eye]

			right_shoulder = points[5][:2]
			right_shoulder = [int(w) for w in right_shoulder]

			left_shoulder = points[11][:2]  # left leg
			left_shoulder = [int(w) for w in left_shoulder]

			cv2.circle(frame, center=tuple(right_shoulder), radius=4, color=(0, 255, 225), thickness=6)
			cv2.circle(frame, center=tuple(left_shoulder), radius=4, color=(225, 0, 225), thickness=6)
			
			# Compute the distance between (x1, y1, x2, y2)
			x1, y1 = right_shoulder[0], right_shoulder[1]
			x2, y2 = left_shoulder[0], left_shoulder[1]
			x3, y3 = eye[0], eye[1]



			distance = euclidean_distance(x1, y1, x2, y2)
			print(distance)
			(idx, owner)  = recognize_seat_owner(frame, box)
			if distance > 170:
				res2 = "Adult "+owner
				
			else:
				res2 ="Child " + owner

			cv2.putText(frame, res2 , (x3,y3), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
			
			if idx is not None:
				occupancy_flags[idx] = True





		# =======================================================================================================
		print(occupancy_flags)

		# Show a frame
		cv2.imshow('Frame', frame)

		# Quit the process
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Destroy all the windows
	cap.release()
	cv2.destroyAllWindows()
