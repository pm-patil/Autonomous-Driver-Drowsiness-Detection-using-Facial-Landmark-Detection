from imutils import face_utils # For array to Numpy array
import imutils # video feed cropping
import dlib # Face landmark predictor
import cv2 # feed conversion to greyscale
import eye # module1    - eye.py
import mouth # module2  - mouth.py

	
	

frame_check = 20	   # counter for 20 consecutive frames        (constant)  (customizable)
thresh = 0.26   # Eye     # Eye spect ratio                          (threshold) (customizable)             
thresh2 = 0.55  # Mouth   # Verticle/horizontal ratio for open mouth (threshold) (customizable)
eye_t=0

	
detect = dlib.get_frontal_face_detector() # initialize 
predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")# Import pre-trained model 

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]             # 2D array of 6 points (x,y - co-ordinates) 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]            # 2D array of 6 points
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]                  # 2D array of 17 points
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]                 # 2D array of 9 points
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]                # 2D array of 20 points (inner-outer mouth)
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]       # 2D array of 5 points
(riStart, riEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]      # 2D array of 5 points ---------(total 68 points)
cap=cv2.VideoCapture(0) # starts camera /video feed 

flag=0                                                                       # flag for eye   (number of frames for which eyes are closed/drowsy )
flag2=0                                                                      # flag for mouth (number of frames for which mouth is open )
eye_t=0
eye_t1=0
cnt=0
while True:                                                                  # loop until 'q' is pressed to exit loop
	ret, frame=cap.read()                                                # opencv  - true if farme is present
	frame = imutils.resize(frame, width=450)                             # resize feed to small window 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                       # frame is converted to greyscale  
	subjects = detect(gray, 0)                                           #opencv2 detect face from frame(greyscale)
	
		
	for subject in subjects:
		shape = predict(gray, subject)                               # predict points -  68 points
		shape = face_utils.shape_to_np(shape)                        # converting to NumPy Array
		cnt=cnt+1
		leftEye = shape[lStart:lEnd]                                 # 2D array - first point - lstart with x,y 
		rightEye = shape[rStart:rEnd]                                #
		leftEyeb = shape[leStart:leEnd]                              #
		rightEyeb = shape[riStart:riEnd]                             # CREATE  7  2D-Arrays (realtime - per frame) 
		jaws = shape[jStart:jEnd]#                                   #
		oneNose = shape[nStart:nEnd]#                                #
		#                                 #
		oneNose1 = oneNose[0:4]
		noseHulli = oneNose[3:nEnd]
		oneMouth = shape[mStart:mEnd]                                # 2D array - first point - mstart with x,y
		oneMouthi = oneMouth[12:mEnd]
		
		leftEAR = eye.eye_aspect_ratio(leftEye)                      # EAR for lefteye (vertical /horizontal) from eye package
		rightEAR = eye.eye_aspect_ratio(rightEye)                    # EAR for righteye (vertical /horizontal) 
		ear = (leftEAR + rightEAR) / 2.0                             # average for accuracy
		yawnRatio = mouth.yawning(oneMouth)                          # ratio of mouth (vertical /horizontal)  from mouth package
		eye_t=eye_t+ear
		if cnt==200:
			
			eye_t1=eye_t/cnt
			#print(eye_t1,ear,cnt)
			eye_t1=eye_t1-0.05
			#print(eye_t1,ear,cnt)
			#print(oneNose1)
		leftEyeHull = cv2.convexHull(leftEye)                        # create convex shape by joining points in leftEye array
		rightEyeHull = cv2.convexHull(rightEye)                      #
		noseHull = cv2.convexHull(oneNose)#                          #
		noseHull1 = cv2.convexHull(noseHulli)#			       #
		
		mouthHull = cv2.convexHull(oneMouth)#                        # create convex shape by joining points in oneMouth array
		mouthHulli = cv2.convexHull(oneMouthi) 
		 
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)                   # DRAW shape on live feed(frame) - polygon
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)                  #
		#cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)#                    #
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)                     # DRAW shape on live feed(frame) - polygon
		#jawsHull = cv2.polylines(frame,[jaws],False,(0, 255, 0), 1 )#               # CREATE and DRAW shape - line
		leftEyebHull = cv2.polylines(frame,[leftEyeb],False,(0, 255, 0), 1 )#       # CREATE and DRAW shape - line
		noselines= cv2.polylines(frame,[oneNose1],False,(0, 255, 0), 1 )
		rightEyebHull = cv2.polylines(frame,[rightEyeb],False,(0, 255, 0), 1 )#     # CREATE and DRAW shape - line 
		cv2.drawContours(frame, [mouthHulli], -1, (0, 255, 0), 1) 
		cv2.drawContours(frame, [noseHull1], -1, (0, 255, 0), 1) 
		
		
		######################################       LOGIC      ##########################
		
		if ear < eye_t1:                                                                             # if eye ratio below threshold i.e drowsy
			
			flag += 1                                                                            # increase flag (per frame)  
			#print (flag)                                                                        # for debug
			if flag >= frame_check:                                                               # if 20(frame_check) consecutive times flag is increased 
				cv2.putText(frame, "****************ALERT EYE!****************", (10, 30),    # put live text on video feed (per frame) (top)
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                        
				cv2.putText(frame, "****************ALERT EYE!****************", (10,325),    # put live text on video feed (per frame) (bottom)
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                        
		else:
			flag = 0                                                                              # if driver is awake set flag to 0 
			
			
			
			
			
			
		if yawnRatio > thresh2:                                                                        # if mouth ratio below threshold i.e drowsy
			                         
			flag2 += 1                                                                             # increase flag2 (per frame)
			#print (flag2)                                                                         # for debug
			if flag2 >= frame_check:                                                               # if 20(frame_check) consecutive times flag2 is increased 
				cv2.putText(frame, "****************ALERT YAWN!****************", (10, 30),    # put live text on video feed (per frame) (top) 
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT YAWN!****************", (10,325),     # put live text on video feed (per frame) (bottom)
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag2 = 0                                                                               # if driver is awake set flag2 to 0 
			
			
		#######################################      LOGIC END      ############################	
			
	        
	cv2.imshow("Drowsy Driver", frame)            # show LIVE FEED WINDOW 
	key = cv2.waitKey(1) & 0xFF           # check for keypress
	if key == ord("q"):                   # if key ==q
		break                         #  break
cv2.destroyAllWindows()                       # if while loop breaks delete all LIVE FEED windows  

