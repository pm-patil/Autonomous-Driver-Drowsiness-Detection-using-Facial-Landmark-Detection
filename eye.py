from scipy.spatial import distance
	
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5]) # vertical 
	B = distance.euclidean(eye[2], eye[4]) # vertical
	C = distance.euclidean(eye[0], eye[3]) # Horizontal
	ear = (A + B) / (2.0 * C)
	return ear # Eye Aspect Ratio	
