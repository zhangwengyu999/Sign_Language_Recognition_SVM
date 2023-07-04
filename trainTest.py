#############################################################################
# Sign Language Recognition System (SLRS)                                   #
#                                                                           #
CAM_NO = 2 # e.g. = 0 or 1 or 2. Make sure your camera works properly;      #
#                                                                           #
#   *********************************************************************   #
#   ** Please carefully read README.md file first before running **   #
#   *********************************************************************   #
#                                                                           #
#############################################################################

import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
from gtts import gTTS
from playsound import playsound

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# detect hand with mediapipe, return mediapipe hand results
def handDetect(image):
    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
        # Convert the BGR image to RGB before processing.
        results = hands.process(image)
        if not results.multi_hand_landmarks:
            return None
        else:
            return results

# draw and get landmarks and connections
def getLandmarks(image, results, show):
    thickness = 5
    connections = [[4, 3, 2, 1, 0],# thumb
               [8, 7, 6, 5],# index
               [12, 11, 10, 9],# middle
               [16, 15, 14, 13],# ring
               [20, 19, 18, 17, 0], #pinky
               [3, 5, 9, 13, 17]# palm
               ]
    for hand_landmarks in results.multi_hand_landmarks:
        h, w, c = image.shape
        id2cords = {}
        for i in range(0,21):
            idx, ftx, fty = i, int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)
            id2cords[idx] = [ftx, fty]
        if show:
            for line in connections:
                pts = [[id2cords[idx][0], id2cords[idx][1]] for idx in line]
                pts = np.array(pts, np.int32)
                pts = pts.reshape((-1, 1, 2))
                image = cv2.polylines(image, [pts], False, (0, 255, 65), thickness)
            for idx in id2cords:
                image = cv2.circle(
                    image, (id2cords[idx][0], id2cords[idx][1]), 10, (0, 255, 65), thickness)
            image = cv2.circle(
                image, (id2cords[0][0], id2cords[0][1]), 15, (0, 0, 0), thickness)
        return image, id2cords

# get minimum square bounding box, return image, top-left x, top-left y, edge length
def getBoundingBox(image, id2cords):
    minX, minY, maxX, maxY = 100000, 100000, 0, 0
    for idx in id2cords:
        if id2cords[idx][0] < minX:
            minX = id2cords[idx][0]
        if id2cords[idx][0] > maxX:
            maxX = id2cords[idx][0]
        if id2cords[idx][1] < minY:
            minY = id2cords[idx][1]
        if id2cords[idx][1] > maxY:
            maxY = id2cords[idx][1]
    maxLen = max(maxX-minX, maxY-minY)
    if (maxX-minX < maxY-minY):
        minX = minX-int((maxLen-(maxX-minX))/2)
    else:
        minY = minY-int((maxLen-(maxY-minY))/2)
    # draw square bounding box
    image = cv2.rectangle(image, (minX-25, minY-25), (minX+maxLen+25, minY+maxLen+25), (0, 255, 65), 4) 
    return image, minX-25, minY-25, maxLen+25

# get fusion feature
def getFusionFeature(inX, inY, inLen,inImage, id2cords):
    img_gray = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY) # grey scale
    # Get mask based on landmarks
    thickness = 60
    connections = [[4, 3, 2, 1, 0],# thumb
               [8, 7, 6, 5],# index
               [12, 11, 10, 9],# middle
               [16, 15, 14, 13],# ring
               [20, 19, 18, 17, 0], #pinky
               [3, 5, 9, 13, 17]# palm
               ]
    mask = np.zeros(inImage.shape[:2], dtype = np.uint8) # set a black mask

    for line in connections:
        pts = [[id2cords[idx][0], id2cords[idx][1]] for idx in line]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [pts], False, (255, 255, 255), thickness) # draw white lines on mask
    
    # Get fusion image with pixel and mask
    img_fusion_color = cv2.bitwise_and(inImage, inImage, mask=mask) # get color pixel on hand only for display
    img_fusion = cv2.bitwise_and(img_gray, img_gray, mask=mask) # get pixel on hand only
    img_fusion_square = img_fusion[inY:inY+inLen, inX:inX+inLen] # get AOI pixels as feature
    img_fusion_28 = cv2.resize(img_fusion_square, (28,28), interpolation = cv2.INTER_AREA) # resize to 28*28
    feat = img_fusion_28.reshape(1,-1)/255 # normalize by 255 to 0-1
    return img_fusion_color, feat[0]

# build SVM models based on features
def getModel():
    # load sample data from files
    # load sample data based on fusion feature
    x,y=np.load('fusion_feat_x.npy'),np.load('fusion_feat_y.npy')
    x = x.reshape(x.shape[0],-1)
    feat_x, X_test, feat_y, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8 ,random_state = 10)
    model = svm.SVC(kernel='linear') # Apply SVM
    print('Fitting, please wait...')
    model.fit(feat_x,feat_y) # Fit the model
    print('Fitting done!')
    results_test = model.predict(X_test) # make prediction on test data
    print('Accuracy calculating, please wait...')
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=results_test) # calculate accuracy
    print("Accuracy:", accuracy)
    print('Cross validating, please wait...')
    val_scores = cross_val_score(model, feat_x, feat_y, cv=5) # cross validation
    print('Cross validation scores: ',val_scores)
    
    return model, accuracy, val_scores

def main():
    # Train and test model
    model, accuracy, val = getModel()
    
    # Camera input
    cap = cv2.VideoCapture(CAM_NO)
    windName = "Sign Language Recognition System"
    cv2.namedWindow(windName)
    cv2.resizeWindow(windName, 1024, 768)
    landmarks_on = False # draw landmarks or not
    speaker_on = False # speak out sign or not
    fusionImg_on = False # draw fusion image or not
    
    while True:
        success, image = cap.read()
        image_box = image.copy()
        mx = 0
        my = 0
        ml = 0
        label = None
        sign = ""
        
        # Hand detection
        if not success:
            print("Open cam failed ....")
            continue
        image = cv2.flip(image, 1)
        image_box = cv2.flip(image_box, 1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = handDetect(imageRGB)
        if results == None:
            print("Nothing detected ...")
        else:
            image,id2cords = getLandmarks(image, results, landmarks_on)
            image,mx,my,ml = getBoundingBox(image, id2cords)
        
        # Hand detected, then extract features
        if (mx>=0 and my >=0 and ml>0 and (mx+ml+50)<image.shape[1] and (my+ml+50)<image.shape[0]):
            fusionImg,fusionFeat = getFusionFeature(mx,my,ml,image,id2cords) # get fusion image and feature
            label=model.predict(np.array(fusionFeat).reshape(1,-1))[0] # predict label
            if fusionImg_on:
                image = fusionImg
            
            sign = chr(label+64) # convert label to sign
            print('label=',label,chr(label+64))
        else:
            sign = "Not Detected"
            print("no feat ...")
        
        # Display image and information
        image=cv2.rectangle(image, (10,10),(1000,210),color=(100, 100, 100),thickness=-1)
        image=cv2.putText(image,"[0] Exit; [1] Landmark; [2] Fusion; [3] Speaker",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0, 255, 65),3)
        image=cv2.putText(image,"Accuracy: "+str(accuracy.round(2)), (10,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0, 255, 65),3)
        image=cv2.putText(image,"Recognized Sign: ",(10,150),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0, 255, 65),3)
        image=cv2.putText(image,sign,(350,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0, 255, 65),4)
        image=cv2.putText(image,"Made by Mike_ZHANG",(image.shape[1]-350,image.shape[0]-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 65),2)
            
        cv2.imshow(windName, image)
        
        if speaker_on and sign != "-":
            audio = gTTS(text=sign, lang='en', slow=False)
            audio.save("sign.mp3")
            playsound("sign.mp3")

        # keyboard cmd input
        key=cv2.waitKey(1) & 0xFF
        # 0 for exit
        if key == ord('0') or key==27:
            break
        # 1 for toggle landmarks
        if key == ord('1'):
            landmarks_on=not landmarks_on
        # 2 for toggle fusion image
        if key == ord('2'):
            fusionImg_on=not fusionImg_on
        # 3 for toggle speaker
        if key == ord('3'):
            speaker_on=not speaker_on
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # train, validation, test model and recognize sign;
    main()
