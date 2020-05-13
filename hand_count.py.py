import cv2
import numpy as np
import math


Use below Command to use webcam
cam = cv2.VideoCapture(0)

# Lower Bound and upper bound to detect skin color
lowerBound = np.array([0, 30, 60])
upperBound = np.array([30, 150, 255])
   
# kernels for Morphological Transformation
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((3, 3))



while True:
    ret, img2 = cam.read()
    img2 = cv2.flip(img2,0)
    #img2=cv2.GaussianBlur(src=img2, ksize=(11,11)
    #                    ,sigmaX=1000,sigmaY=3)
    #img2=cv2.blur(img2,(1000,1000));
    #img4=cv2.medianBlur(img2,9)
    
    # Applying the bilateral Filter 
    # of diameter 5 sigmaColor=60, sigmaSpace=60 
    img3 = cv2.bilateralFilter(img2, 5 , 60, 60)
    
    # Converting to HSV format
    imgHSV = cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
    
    # Selecting the skin color fromgthe image 
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    
    # Applying Morphological Transfomaton on this 
    # binary image
    erosion=cv2.erode(mask,kernelOpen,5)
    maskOpen = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernelOpen)
    
    # nes is only  for display purpose It tells the position of hand 
    # wrt orignal image
    nes = cv2.bitwise_and(img3, img3,mask=erosion)
    
    # Finding contours
    _, contours, heirarchy = cv2.findContours(maskOpen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours is a list of list of all the contours , Each list contain a
    # list of continous cordinates of that contour
    length = len(contours)
    maxArea = -1
    if length > 0:
        # Finding contour with max area 
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        # ci is the index of contour with max area
        # res is the list of coordinates of the contour with max area
        res = contours[ci]
        
        # The next three lines are for drawing the contours and its Hull
        hull1 = cv2.convexHull(res, True)
        cv2.drawContours(img2, [res], 0, (0, 255, 0), 1)
        cv2.drawContours(img2, [hull1], 0, (0, 0, 255), 3)
       
        # hull is the hull wrt the contour res
        hull = cv2.convexHull(res, returnPoints=False)
        
        # calculating the centroid of the hand i.e. given contour
        M = cv2.moments(res)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.circle(nes, (cX, cY), 30, (255, 0, 0), 1)
        
        # If the hull polygon has more than 3 points
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            # defects is a (N x 1 x 4) shaped array where 
            # N is the number of sets of these defects - each defect consist of 
            # start ,end ,far coordinates and farthest distance
            if type(defects) != type(None):  # avoid crashing.   (BUG not found)
                cnt = 1
                # cnt is the number of fingures
                # calculate the angle between fingures and checking if its less
                # then 90 degree , start and end represents the tip of two fingures
                for i in range(defects.shape[0]):  
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    tArea= 0.5 * math.fabs( (start[0]-far[0])*(end[1]-far[1])-(end[0]-far[0])*(start[1]-far[1]) )
                      # angle less than 90 degree, treat as fingers
                    # if angle <= math.pi / 2 and tArea>1000 and start[1]<cY and end[1]<cY and d>30:
                    if angle <= math.pi / 2 and start[1]<cY and end[1]<cY:
                        cnt += 1
                        cv2.circle(nes, start, 8, [211, 84, 0], 1)
                        cv2.circle(nes, end, 8, [211, 84, 0], 1)
                cv2.putText(nes, str(cnt), (20, 420), cv2.FONT_HERSHEY_SIMPLEX,6, (255, 255, 255), 2)
                print(cnt)
        
        

    
    
    #img5 = cv2.hconcat([erosion,nes])
    cv2.imshow("Output", nes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
























