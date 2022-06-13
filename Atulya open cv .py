import cv2
import cv2.aruco as aruco
import math
import numpy as np

def findaruco(img):#This function returns the cornera nd id of the given aruco marker
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()

    (corners , ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    return corners, ids

def colour(value): #This function takes the BGR values of the pixel as input and returns the corresponding id of the color 
    if(value[0]==0) and (value[1]==0) and (value[2]==0):
        return "3"
    elif (value[0]==210) and (value[1]==222) and (value[2]==228) :
        return "4"
    elif (value[0]==9) and (value[1]==127) and (value[2]==240) :
        return "2"
    elif (value[0]== 79) and (value[1] ==209) and (value[2]==146):
        return "1"
    else:
        return None

markers = {}  #Dictonary for storing aruco markers
for i in ["Ha", "HaHa","LMAO","XD"]:
    img = cv2.imread(f"Images//{i}.jpg")
    a, b= findaruco(img) #The corners and the id of the aruco markers is extracted
    slope = (a[0][0][1][1]- a[0][0][0][1])/(a[0][0][1][0]- a[0][0][0][0]) 
    angle = math.degrees(math.atan(slope))  #Angle at which aruco marker is present from normal position is calculated
    cx  = int((int(a[0][0][0][0])+int(a[0][0][2][0]))/2) #x coordinate of center of aruco marker
    cy  = int((int(a[0][0][0][1])+int(a[0][0][2][1]))/2) #y coordinate of center of aruco marker
    (h, w) = img.shape[:2]
    rotate = cv2.getRotationMatrix2D((cx,cy),angle,1.0)  #Rotation matrix for rotating the aruco marker 
    rotated = cv2.warpAffine(img,rotate,(w,h)) #The aruco marker is rotated to normal position
    c,d = findaruco(rotated) #The corners and ids of aruco are again extracted
    markers[str(b[0][0])] = rotated[int(c[0][0][0][0]):int(c[0][0][2][0]),int(c[0][0][0][1]):int(c[0][0][2][1])] #The aruco marker is stored with corresponding id as key

for i in markers.keys(): #Arco markers after rotation are displayed
    cv2.imshow(f"{i}",markers[i])
key = cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread("Images//CVtask.jpg") #The image given to perform the task is read
img2 = cv2.resize(img2, (877,620))
cv2.imshow("Test",img2)
k2 = cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)#The image is converted to grey
canny = cv2.Canny(gray,30,150) #The edges of image are detected
cv2.imshow("Canny",canny)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

img3 = img2.copy()
squares = [] #List that will store the attributes of the squares
c, h = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Contours are found for the canny image
for cont in c:
    approx = cv2.approxPolyDP(cont, 0.01* cv2.arcLength(cont, True), True) #The corners are calculated for each set of contours detected for a shape
    if len(approx)== 4:
        x, y, w, h = cv2.boundingRect(approx) #The x - x coordinate, y- ycoordinate , w-width, h-height of the bounding rectangle is calculated
        ratio = float(w)/h #The ratio fo width and height of the bounding rectangle is calculated 
        if (ratio> 0.95) and (ratio < 1.05) : #For a square the ratio should be near to 1
            cv2.drawContours(img3,[approx],0,(0,0,255))
            ltx, lty = approx[0][0][0], approx[0][0][1] #x and y coordinate of left top corner is calculated
            rtx, rty = approx[1][0][0], approx[1][0][1] #x and y coordinate of right top corner is calculated
            rbx, rby = approx[2][0][0], approx[2][0][1] #x and y coordinate of right bottom corner is calculated
            lbx, lby = approx[3][0][0], approx[3][0][1] #x and y coordinate of left bottom corner is calculated
            #circles are drawn at all corners for visualization of corners
            cv2.circle(img3, (ltx,lty), 5, (0,0,255),-1)
            cv2.circle(img3, (rtx,rty), 5, (0,0,255),-1)
            cv2.circle(img3, (lbx,lby), 5, (0,0,255),-1)
            cv2.circle(img3, (rbx,rby), 5, (0,0,255),-1)
            cenx = int((ltx + rbx)/2) #x coordinate of the center of square is calculated
            ceny = int((lty + rty)/2) #y coordinate of the center of square is calculated
            angle = float(rty - lty)/(rtx - ltx) #slope of the angle of side of square from horizonatal is calculated
            angle = math.degrees(math.atan(angle)) #Angle fo square from horizontal is calculated from a 
            squares.append([[ltx,lty],[rtx,rty],[rbx,rby],[lbx,lby],[cenx,ceny],angle]) #all the Attributes of the squares is stored in a list 'squares'
            id = colour(img3[ceny][cenx])
cv2.imshow("Im",img3) #The image with countours and circles at corners is displayed
key = cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(4):
    im_1 = np.zeros([1300,1300,3],dtype =np.uint8) #A blank matrix for is formed of dimensions 1300x1300
    im_1.fill(255) #The matrix is filled with 255 so that the resulting image is a white image
    im_1[260:880,130:1007] = img2 #A part of the above image is replaced by values of our orignal image
    #By the above operations we get a image of our orignal image which is padded by white pixels to avoid and loss of the parts of the image in rotation  
    n = squares[2*i][5]
    rotate = cv2.getRotationMatrix2D((542,542),n,1.0) #Rotation matrix for rotation of the above image to bring the one side of the square in line with horizontal
    rotated = cv2.warpAffine(im_1,rotate,(1300,1300)) #The image is rotated and one of the squares has sides paralled to axis
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,0,35)
    rotated1 = rotated.copy()
    c1, h1 = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Contours are found the rotated image
    for cont in c1:
        approx1 = cv2.approxPolyDP(cont, 0.01* cv2.arcLength(cont, True), True) #Corners are detected for the contours
        if len(approx1)== 4:
            x, y, w, h = cv2.boundingRect(approx1) #The x - x coordinate, y- ycoordinate , w-width, h-height of the bounding rectangle is calculated
            ratio = float(w)/h #The ratio fo width and height of the bounding rectangle is calculated 
            if (ratio> 0.99) and (ratio < 1.05) : #For a square the ratio should be near to 1
                ltx, lty = approx1[0][0][0], approx1[0][0][1] #x and y coordinate of left top corner is calculated
                rtx = approx1[1][0][0] #x coordinate of the right top corner is calculated
                rbx, rby = approx1[2][0][0], approx1[2][0][1] #x and y coordinate of left bottom corner is calculated
                cenx = int((ltx + rbx)/2) #x coordinate of the center of square is calculated
                ceny = int((lty + rby)/2) #y coordinate of the center of square is calculated
                id = colour(rotated1[ceny][cenx]) #ID of the aruco marker is found for BGR values of the center pixel
                ratio1 = abs(float(ltx-rtx)/w) #The ratio of the length of the square and the width of bounding rectangle is calculated
                if (id != None) and (ratio1 < 1.05) and (ratio1> 0.95):
                    #IF the ratio is near to one it means the bounding rectangle and the square are in same orientaion and we will replace the square by aruco in that case
                    aru = markers[id] # aruco marker for the correspoding id is taken
                    aru = cv2.resize(aru,(w,h)) #aruco marker is rsized to fit in the square
                    rotated1[y:y+h,x:x+w] = aru #aruco marker is replaced in place of the square
                    rotate = cv2.getRotationMatrix2D((542,542),-n,1.0) #The rotaion matrix to rotate the image to orignal orientation 
                    rotated1 = cv2.warpAffine(rotated1,rotate,(1300,1300)) #The image is rotated to orignal orientation 
                    img2 = rotated1[260:880,130:1007] #The padding is removed
                    cv2.imshow("Aruco is pasted !!",img2) #The image is displayed after pasting each aruco marker
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    break

cv2.imwrite("final.jpg",img2) #The image is rendered to a jpg file