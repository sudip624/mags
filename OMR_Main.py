import cv2
import numpy as np
import utils

###############PARAMETER######################
path = "omr_img1.jpg"
widthImg = 580
heightImg = 650
questions = 5
choices = 5
# Mention which are the actual answers in array
# Each index in array represents each correct answer circle of the row in OMR
ans = [1,2,0,1,4]
##############################################

img = cv2.imread(path)

# PREPPROCESSING
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()

#Grayscale Image
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Blurry Image
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
#Canny Image
imgCanny = cv2.Canny(imgBlur,10,50)

#Finding All contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

#Find Rectangles
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
# print(biggestContour)
# print(biggestContour.shape)
gradePoints = utils.getCornerPoints(rectCon[1])

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    # Separate Grade Section
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    # cv2.imshow("Grade", imgGradeDisplay)

    # The bubbles that not having marking, will have less amnt have non-zero pixels vice-versa
    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray,150,255,cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThresh)
    # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

    # GETTING NON ZERO PIXEL VALUES OF EACH BOX
    myPixelVal = np.zeros((questions,choices))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices) : countR += 1; countC = 0
    # print(myPixelVal)

    # FINDING THE USER ANSWERS AND PUTTING THEM IN A LIST
    myIndex = []
    for x in range (0,questions):
        arr = myPixelVal[x]
        # print("arr", arr)
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    # print(myIndex)

    # GRADING
    grading = []
    for x in range (0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else: grading.append(0)
    # print(grading)
    score = (sum(grading)/questions) * 100
    print(score)

    # DISPLAYING ANSWERS
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))



imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray,imgBlur,imgCanny],
              [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
              [imgResult, imgRawDrawing, imgInvWarp, imgBlank])
imgStacked = utils.stackImages(imageArray,0.3)

cv2.imshow("Stacked Image", imgStacked)
cv2.waitKey(0)                     