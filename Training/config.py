

POSITIVE_THRESHOLD = 2000.0


TRAINING_FILE = 'training.xml'


POSITIVE_DIR = './training/positive'



POSITIVE_LABEL = 1
FACE_WIDTH  = 92
FACE_HEIGHT = 112


HAAR_FACES         = 'haarcascade_frontalface_alt.xml'
HAAR_SCALE_FACTOR  = 1.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (30, 30)


DEBUG_IMAGE = 'capture.png'

def get_camera():	
	
	#import webcam
	#import glob
	#global OpenCVCapture
	#return OpenCVCapture()
	
	import webcam
        return webcam.OpenCVCapture(device_id=0)
