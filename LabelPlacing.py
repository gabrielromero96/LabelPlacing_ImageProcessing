import cv2
import numpy as np
from datetime import datetime


startTime = datetime.now()
cap = cv2.VideoCapture(0)

# Variables to play with
thresholdLine1 = 87 # [138]
thresholdLine2 = 116 # [220]
areaMin = 10000
areaMax = 500000
Xmin1, Ymin1, Xmin2, Ymin2, = 185, 120, 465, 330
Xmax1, Ymax1, Xmax2, Ymax2, = Xmin1 - 40, Ymin1 - 40, Xmin2 + 40, Ymin2 + 40

"""
# Read logo and resize
logo = cv2.imread('WispShapeRS.jpg')
size = 1200; scale = 2.3; width = 235; height = 150  # Image has a 1.56578 width/height ratio
dimension = (int(width*scale), int(height*scale))
logo = cv2.resize(logo, dimension)
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY_INV)
"""


def getContours(img, imgContour):  # This function is to highlight the contours

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.circle(imgContour, (40, 40), 30, (0, 0, 255), -1)
    cv2.circle(imgContour, (40, 40), 30, (0, 0, 0), 2)


    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area >= areaMin and area <= areaMax:  # Show only contour with an area greater than value
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)  # Store perimeter length. True is to confirm contour is closed
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # Detect figure based how many vertices are present
            # print(len(approx)) # Prints how may vertices are present. For square/rectangles should be 4
            x, y, w, h = approx[1][0][0], approx[1][0][1], approx[3][0][0], approx[3][0][1] # Coordinates for label perimeter
            if x > w or y > h: # If coordinates in wrong order
                x, y, w, h = approx[0][0][0], approx[0][0][1], approx[2][0][0], approx[2][0][1]
            bbx, bby, bbw, bbh = cv2.boundingRect(approx)  # first corner coordinates, width and length, for bounding box

            # Checking if rectangle present and inside tolerance
            if (len(approx) == 4 and Xmax1 <= x <= Xmin1 and Ymax1 <= y <= Ymin1
                    and Xmax2 >= w >= Xmin2 and Ymax2 >= h >= Ymin2):
                passLabel = "PLACED"
                cv2.circle(imgContour, (40, 40), 29, (0, 255, 0), -1)
            else:
                passLabel = "MISSPLACED"

            # Drawing bounding box, corners and pass info
            # cv2.rectangle(imgContour, (bbx, bby), (bbx + bbw, bby + bbh), (0, 255, 0), 3)
            cv2.putText(imgContour, passLabel,(w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 1)
            cv2.circle(imgContour, (x, y), 5, (255, 255, 0), -1)
            cv2.circle(imgContour, (w, h), 5, (255, 255, 0), -1)



while True:

    """
    # Region of Image (ROI), where we want to insert logo 
    roi = img[-size + 100: -100, -size + 300: -300]

    # Set an index of where the mask is 
    roi[np.where(mask)] = 0
    #roi += logo

    # Create mask to filter video with image
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask2 = cv2.threshold(imgHsv, 10, 255, cv2.THRESH_BINARY)
    """

    # Capture frame-by-frame
    success, img = cap.read()

    # Apply blur, canny, and dilatation to obtain contours
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, thresholdLine1, thresholdLine2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)

    # We create a copy from image to prevent red rectangles from interfering line detection
    imgProduct = imgContour.copy()

    # Drawing rectangle for min and max tolerance
    imgProduct = cv2.rectangle(imgProduct, (Xmin1, Ymin1), (Xmin2, Ymin2), (0, 0, 255), 1)
    imgProduct = cv2.rectangle(imgProduct, (Xmax1, Ymax1), (Xmax2, Ymax2), (0, 0, 255), 1)

    cv2.imshow('Label Placing', imgProduct)
    cv2.imshow('Line detection', imgDil)
    # cv2.imshow("Mask2", mask2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture 
cap.release()
cv2.destroyAllWindows() 