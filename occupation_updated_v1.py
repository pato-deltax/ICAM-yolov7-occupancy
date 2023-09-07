import cv2
import torch
import numpy as np
import time
from torchvision import transforms
from models.utils import* 


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


#calculates the Euclidean distance between two points in a two-dimensional space. 
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



# def recognize_seat_owner(frame, box, front_shift=120, back_shift=[50, 100]): # for height and width increase and decrease(40,65)
# 	""" Recognize a seat owner of the passengers """
# 	driver = (800, 450)
# 	front_passenger = (425, 450)
# 	back_middle_passenger =  (600, 275) #[(539, 128), (636, 367)]
# 	back_left_passenger = (690, 275) # [(697, 122), (624, 364)]
# 	back_right_passenger = (510, 275) #[(431, 139), (556, 355)]


def recognize_seat_owner(frame, box, front_shift=150, back_shift=[40, 140]): # for height and width increase and decrease(40,65)
	""" Recognize a seat owner of the passengers """
	driver = (800, 575)
	front_passenger = (425, 565)
	back_middle_passenger = (600, 275)
	back_left_passenger = (690, 275)
	back_right_passenger = (510, 275)

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
	back_middle_passenger_w = (seat[0] - back_shift[0] < back_middle_passenger[0] < seat[0] + back_shift[0])
	back_middle_passenger_h = (seat[1] - back_shift[1]  < back_middle_passenger[1] < seat[1] + back_shift[1])
	back_left_passenger_w = (seat[0] - back_shift[0] < back_left_passenger[0] < seat[0] + back_shift[0])
	back_left_passenger_h = (seat[1] - back_shift[1] < back_left_passenger[1] < seat[1] + back_shift[1])
	back_right_passenger_w = (seat[0] - back_shift[0] < back_right_passenger[0] < seat[0] + back_shift[0])
	back_right_passenger_h = (seat[1] - back_shift[1] < back_right_passenger[1] < seat[1] + back_shift[1])

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
		if type(shift) == int:
			start_point = (center[0] - shift, center[1] - shift)
			end_point = (center[0] + shift, center[1] + shift)
		else:
			start_point = (center[0] - shift[0], center[1] - shift[1])
			end_point = (center[0] + shift[0], center[1] + shift[1])
		cv2.rectangle(frame, start_point, end_point, (255, 255, 255), 1)

	# Define the current seat owner
	if driver_w and driver_h:
		seat_owner = (0, "Driver")
	elif front_passenger_w and front_passenger_h:
		seat_owner = (1, "Front")
	elif back_middle_passenger_w and back_middle_passenger_h:

		seat_owner = (2, "Back middle")
	elif back_left_passenger_w and back_left_passenger_h:
		seat_owner = (3, "Back left")
	elif back_right_passenger_w and back_right_passenger_h:
		seat_owner = (4, "Back right")
	else:
		seat_owner = (None, "Not passenger")
	return seat_owner
print('seat_owner')


if __name__ == '__main__':
	cap = cv2.VideoCapture("/home/shiraz/Downloads/4shiraz/2023-01-12-135839.webm")
	#cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
	prev_frame_time = 0
	new_frame_time = 0

	while (cap.isOpened):
		success, frame = cap.read()
		if not success:
			break

		# Calculating the fps
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = int(fps)
		fps = str(fps)
		cv2.putText(frame, f"FPS: {fps}", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 255), 2, cv2.LINE_8)
	  
		# Get pose landmarks
		frame, bbox = detect_pose_landmarks(frame)

		occupancy_flags = [False, False, False, False, False]
		# =======================================================================================================
		adult_count= 0
		child_count = 0
		true_count= 0
		for box in bbox:
			plot_skeleton_kpts(frame, box[7:].T, 3)
			points = box[7:].T
			points = points.reshape((17,3))

			eye = points[2][:2]
			eye = [int(w) for w in eye]

			right_shoulder = points[6][:2]
			right_shoulder = [int(w) for w in right_shoulder]

			left_shoulder = points[5][:2]
			left_shoulder = [int(w) for w in left_shoulder]
			
			left_Belt = points[11][:2]  # shoulder
			left_Belt = [int(w) for w in left_Belt]

			cv2.circle(frame, center=tuple(right_shoulder), radius=4, color=(0, 0, 0), thickness=6)
			cv2.circle(frame, center=tuple(left_shoulder), radius=4, color=(0, 0, 0), thickness=6)
			cv2.circle(frame, center=tuple(left_Belt), radius=4, color=(0, 0, 0), thickness=6)
		
			# Compute the distance between (x1, y1, x2, y2)
			x1, y1 = right_shoulder[0], right_shoulder[1]
			x2, y2 = left_Belt[0], left_Belt[1]
			x3, y3 = left_shoulder[0], left_shoulder[1]
			

			distance_1 = euclidean_distance(x1, y1, x2, y2)
			distance_2 = euclidean_distance(x1, y1, x3, y3)
			distance = (distance_1+distance_2)/2
			
			(idx, owner)  = recognize_seat_owner(frame, box)
			on_front = ["Driver front", "Front"]
			on_back = ["Back left", "Back middle", "Back right"]

			if owner in on_front:
				if distance > 160:
					res2 = "Adult " + owner 
					adult_count+=1
				else:
					res2 = "Child " + owner 
					child_count+=1
			elif owner in on_back:
				if distance > 90:         # 100 # adult child flipping prefix 80
					res2 = "Adult " + owner 
					adult_count+=1
				else:
					res2 = "Child " + owner 
					child_count+=1
						
			else:
				res2 = owner 
    
			if idx is not None:
				occupancy_flags[idx] = True
    
			true_count = occupancy_flags.count(True)

		count_text = "Total Seat occupied: " + str(true_count)
		cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0)  , 2) # Green color

		cv2.putText(frame, res2, (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

		adult_text = "Total adults: " + str(adult_count)
		cv2.putText(frame, adult_text, (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0)  , 2)

		child_text = "Total children: " + str(child_count)
		cv2.putText(frame, child_text, (10, 110),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0)  , 2)

		# Show a frame
		cv2.imshow('Frame', frame)

		# Quit the process
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Destroy all the windows
	cap.release()
	cv2.destroyAllWindows()
