import cv2
import numpy as np
import sys
import math
from skimage.feature import hog

def gradient_energy(img_gray):
	sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=-1)
	sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=-1)
	magnitude,_= cv2.cartToPolar(sobelx, sobely)
	abs_mag = cv2.normalize(magnitude, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	return abs_mag.astype('uint8')

def calc_entropy(img):
	hist = cv2.calcHist([img], [0], None, [256], [0, 255])
	hist[hist==0]=0.0000001
	prob=hist/(img.shape[0] * img.shape[1])
	entropy=np.sum(-1*prob*np.log10(prob)/np.log10(2))
	return entropy

def entropy_energy(img_gray):
	fin_image = cv2.copyMakeBorder(img_gray,8,8,8,8,cv2.BORDER_REFLECT)
	entropy = np.zeros(fin_image.shape)
	for i in range(0,img_gray.shape[0]+9):
		for j in range(0,img_gray.shape[1]+9):
			entropy[i,j] = calc_entropy(fin_image[i:i+9,j:j+9])
	e1_energy = gradient_energy(img_gray)
	entropy_roi = entropy[8:fin_image.shape[0]-8,8:fin_image.shape[1]-8]
	entropy_fin=entropy_roi + e1_energy
	return entropy_fin.astype('uint8')

def gradient_hog_energy(img_gray):
	fd, hog_img = hog(img_gray, orientations=8, pixels_per_cell=(11, 11), cells_per_block=(1, 1), visualize=True, multichannel=False)
	hog_e1 = gradient_energy(img_gray)/np.max(hog_img)
	hog_e1=cv2.normalize(hog_e1, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	return hog_e1.astype('uint8')

def mask_energy(img_gray,mask):
	energy=gradient_energy(img_gray)
	modified_energy = energy.copy().astype(np.float32)
	for i in range(img_gray.shape[0]):
		for j in range(img_gray.shape[1]):
			if mask[i,j]!=0:
				modified_energy[i,j]=-10000
	abs_mag = cv2.normalize(modified_energy, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	return abs_mag.astype(np.float32)

def energy_of_image(img_gray,energy=3,mask=None):
	e=None
	if energy==1:
		e=gradient_energy(img_gray)

	elif energy==2:
		e=entropy_energy(img_gray)

	elif energy==3:
		e=gradient_hog_energy(img_gray)

	elif energy==4:
		e=mask_energy(img_gray,mask)

	return e



if __name__=="__main__":
	image1=cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
	energy_func=sys.argv[2]
	d=0

	if energy_func=="e1_gradient" :
		d=1
	elif energy_func=="entropy" :
		d=2
	elif energy_func=="e1_hog":
		d=3

	image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	img1=energy_of_image(image1,d)
	cv2.imwrite(energy_func+".jpeg",img1)
	# cv2.imshow("e", img1)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



