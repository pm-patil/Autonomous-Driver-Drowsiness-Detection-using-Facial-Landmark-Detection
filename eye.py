from scipy.spatial import distance                                 # distance function
	
def eye_aspect_ratio(eye):                                         # called from main.py for each frame for left/right eye
	A = distance.euclidean(eye[1], eye[5]) # vertical          # distance between vertical points - 1st 
	B = distance.euclidean(eye[2], eye[4]) # vertical          # distance between vertical points - 2nd
	C = distance.euclidean(eye[0], eye[3]) # Horizontal        # distance between horizontal points
	ear = (A + B) / (2.0 * C)                                  # average for accuracy and find vertical /horizontal ratio (EAR)
	return ear # Eye Aspect Ratio	                            # EAR for left/right eye
