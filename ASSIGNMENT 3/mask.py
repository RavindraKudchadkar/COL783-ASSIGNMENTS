import cv2
import numpy as np
import sys

def show_img(title,img,wait):
    cv2.imshow(title, img)
    k = cv2.waitKey(wait)
    cv2.destroyWindow(title)
    return k

class select_points_for_mask:
    def __init__(self,feature_points,image):
        print("Select the 30 points")
        cv2.namedWindow('Select_Points')
        cv2.setMouseCallback('Select_Points', self.select_point)
        self.points = np.zeros((15, 2), dtype="int")
        self.count = 0
        self.inputFeature = 0
        self.output_marked_image = np.copy(image)
        for (x, y) in feature_points:
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
        while self.count != 15:
            cv2.imshow("Select_Points", self.output_marked_image)
            cv2.waitKey(20)
        cv2.imshow("Select_Points", self.output_marked_image)
        cv2.waitKey(20)
        cv2.destroyWindow("Select_Points")

    def getOutputMarkedImage(self):
        return self.output_marked_image

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
            self.points[self.inputFeature + self.count] = np.array([x, y], dtype="int")
            self.count = self.count + 1
        if self.count == 15:
            print("Points successfully selected")

def feature_points_detection(img):
    
    img_color = img.copy()
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)


    points = []

    fore = select_points_for_mask(points, img_color)
    points = fore.points
    points = points.tolist()
    # cv2.imshow("dots", fore.getOutputMarkedImage())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return points

def make_mask(image,n_obj):
	masks=[]
	for i in range(n_obj):
		vertices=feature_points_detection(image.copy())
		mask = np.zeros_like(image.copy()).astype('uint8')
		cv2.fillPoly(mask, np.array([vertices], dtype = np.int32), 255)
		masks.append(mask)

	fin_mask= np.sum(masks,axis=0)
	print(fin_mask.shape)
	return fin_mask.astype('uint8')


# image1=cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
# n_obj=int(sys.argv[2])
# mask=make_mask(image1,n_obj)
# show_img("mask",mask,0)
