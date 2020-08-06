from imutils import face_utils
import imutils
import dlib
import cv2
import eye
import mouth
	
	

frame_check = 20	
thresh = 0.26  #Eye
thresh2 = 0.55  #Mouth


	
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
cap=cv2.VideoCapture(0)

flag=0
flag2=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		jaws = shape[jStart:jEnd]#
		oneNose = shape[nStart:nEnd]#
		oneMouth = shape[mStart:mEnd]
		
		leftEAR = eye.eye_aspect_ratio(leftEye)
		rightEAR = eye.eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		
		yawnRatio = mouth.yawning(oneMouth)
		
		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		noseHull = cv2.convexHull(oneNose)#
		mouthHull = cv2.convexHull(oneMouth)
		
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)#
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		jawsHull = cv2.polylines(frame,[jaws],False,(0, 255, 0), 1 )#
		
		
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT EYE!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT EYE!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
			
		else:
			flag = 0
			
		if yawnRatio > thresh2:
			flag2 += 1
			print (flag2)
			if flag2 >= frame_check:
				cv2.putText(frame, "****************ALERT YAWN!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT YAWN!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag2 = 0
			
			
	        
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()

