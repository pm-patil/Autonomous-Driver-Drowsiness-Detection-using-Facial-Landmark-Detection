from scipy.spatial import distance                                       # distance function

def yawning (oneMouth):                                                  # called from main.py once per frame
	X = distance.euclidean(oneMouth[3], oneMouth[9]) # vertical      # Vertical distance 
	Y = distance.euclidean(oneMouth[0], oneMouth[6]) # Horizontal    # Horizontal distance
	yawn= (X/Y)                                                      # ratio for comparing with thresh2
	return yawn                                                      # return ratio
