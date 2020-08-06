from scipy.spatial import distance

def yawning (oneMouth):
	X = distance.euclidean(oneMouth[3], oneMouth[9]) # vertical
	Y = distance.euclidean(oneMouth[0], oneMouth[6]) # Horizontal	
	yawn= (X/Y)
	return yawn
