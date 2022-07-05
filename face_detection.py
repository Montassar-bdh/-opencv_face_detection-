import cv2, glob

# store all jpg image in a list (images are in folder path)
gimage = glob.glob("*.jpg")

# create a cascade classifier object
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# loop image list
for timage in gimage:
    
    # read the image
    image = cv2.imread(timage)
    
    # scale the image to 50%
    resized = cv2.resize(image, ( int(image.shape[1]/2), int(image.shape[0]/2) ) )
    
    # read image as gray scale image
    grayimage = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    
    # search the coordinates of the the face in the image
    face = detect.detectMultiScale(grayimage,1.05,5)
    
    # draw a rectangle around the face
    for (x,y,w,h) in face:
        cv2.rectangle(resized, (x,y), (x+w,y+h), (0,255,0),2)
    # show the image
    cv2.imshow("image title",resized)
    
    # wait until keyboard key is pressed
    cv2.waitKey(0)
    
    # destroy opened windows
    cv2.destroyAllWindows()
