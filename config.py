import datetime


VIDEO_CONFIG = {
	"VIDEO_CAP": r"C:\Desktop\Crowd-Analysis-main\t1\Town Hall DataCeter3.mp4",

	"IS_CAM" : False,
	"CAM_APPROX_FPS": 3,
	"HIGH_CAM": False,
	"START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}


YOLO_CONFIG = {
	"WEIGHTS_PATH" : "YOLOv4-tiny/yolov4-tiny.weights",
	"CONFIG_PATH" : "YOLOv4-tiny/yolov4-tiny.cfg"
}

SHOW_PROCESSING_OUTPUT = True

SHOW_DETECT = True
DATA_RECORD = True

DATA_RECORD_RATE = 5

RE_CHECK = False

RE_START_TIME = datetime.time(0,0,0) 
RE_END_TIME = datetime.time(23,0,0)
# Check for social distance violation
SD_CHECK = True
# Show violation count
SHOW_VIOLATION_COUNT = True
# Show tracking id
SHOW_TRACKING_ID = True
# Threshold for distance violation
SOCIAL_DISTANCE = 10
# Check for abnormal crowd activity
ABNORMAL_CHECK = True
# Min number of people to check for abnormal
ABNORMAL_MIN_PEOPLE = 5
# Abnormal energy level threshold
ABNORMAL_ENERGY = 1866
# Abnormal activity ratio threhold
ABNORMAL_THRESH = 0.66
# Threshold for human detection minumun confindence
MIN_CONF = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2
# Resize frame for processing
FRAME_SIZE = 1080
# Tracker max missing age before removing (seconds)
TRACK_MAX_AGE = 3
