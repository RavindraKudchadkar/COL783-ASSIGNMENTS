import cv2
import numpy as np
import sys
from seam import *
from time import time

def remove_seam(image,seam):
	fin_image=image.copy()
	mask = np.ones(fin_image.shape,dtype=bool)
	for s in seam:
		fin_image[s[0],s[1]]=[0,0,255]
		mask[s[0],s[1]]=0

	fin_image= fin_image[mask].reshape(fin_image.shape[0],fin_image.shape[1]-1,3)
	return fin_image

def summation(arr,j):
	ans=0
	for i in range(len(arr)):
		ans=ans+arr[i]*2**(i+j)
	return ans

def odd_to_even_size_1(image):
	fin_image=image.copy()
	seam_row,seam_col=[],[]
	if (fin_image.shape[0]%2)!=0:
		seam_row.append(fin_image[-1,:,:])
		fin_image=fin_image[:-1,:,:]
		

	if (fin_image.shape[0]//2)%2!=0:
		seam_row.append(fin_image[-1,:,:])
		seam_row.append(fin_image[-2,:,:])
		fin_image=fin_image[:-2,:,:]

	if (fin_image.shape[1]%2)!=0:
		seam_col.append(fin_image[:,-1,:])
		fin_image=fin_image[:,:-1,:]

	if (fin_image.shape[1]//2)%2!=0:
		seam_col.append(fin_image[:,-1,:])
		seam_col.append(fin_image[:,-2,:])
		fin_image=fin_image[:,:-2,:]

	return fin_image,np.array(seam_row),np.array(seam_col)

def odd_to_even_size_2(image):
	fin_image=image.copy()
	seam_row,seam_col=[],[]
	if (fin_image.shape[1]%2)!=0:
		seam_col.append(fin_image[:,-1,:])
		fin_image=fin_image[:,:-1,:]

	if (fin_image.shape[1]//2)%2!=0:
		seam_col.append(fin_image[:,-1,:])
		seam_col.append(fin_image[:,-2,:])
		fin_image=fin_image[:,:-2,:]

	if (fin_image.shape[0]%2)!=0:
		seam_row.append(fin_image[-1,:,:])
		fin_image=fin_image[:-1,:,:]
		

	if (fin_image.shape[0]//2)%2!=0:
		seam_row.append(fin_image[-1,:,:])
		seam_row.append(fin_image[-2,:,:])
		fin_image=fin_image[:-2,:,:]


	return fin_image,np.array(seam_row),np.array(seam_col)


def fast_seam_removal(image,N,depth,energy=3):
	img1=image.copy()
	if img1.shape[0]%2!=0:
		img1=img1[:-1,:]
	if img1.shape[1]%2!=0:
		img1=img1[:,:-1]

	I=[img1]
	for i in range(depth-1):
		img2=cv2.pyrDown(I[-1])
		I.append(img2)

	n=np.zeros((depth),dtype=np.int64)
	for i in range(depth-1,-1,-1):
		if i==depth-1:
			n[i]=N//2**i
		else:
			n[i]=(N-summation(n[i+1:],i+1))//2**i
			

	print(n)
	masks=[]
	for i in range(depth):
		masks.append(np.ones(I[i].shape,dtype=bool))

	for i in range(len(n)-1,-1,-1):
		for j in range(n[i]):
			im,seam,_=remove_vertical_seam(I[i],energy)
			I[i]=im
			for s in seam:
				masks[i][s[0],s[1]]=0

			for k in range(i-1,-1,-1):
				points=[]
				for y in range(masks[k+1].shape[0]):
					for x in range(masks[k+1].shape[1]):

						if not masks[k+1][y,x].all():
				
							points.append([y,x])
				print("k=",k,"i=",i,"j=",j)
				if k==i-1:
					masks[k+1]=masks[k+1][masks[k+1]].reshape(masks[k+1].shape[0],masks[k+1].shape[1]-1,3)
				else:
					masks[k+1]=masks[k+1][masks[k+1]].reshape(masks[k+1].shape[0],masks[k+1].shape[1]-2*(i-k-1),3)

				points_k=2*np.array(points)

				for p in points_k:
					masks[k][p[0]:p[0]+2,p[1]:p[1]+2]=np.zeros((2,2,3),dtype=bool)
					
				I[k]=I[k][masks[k]].reshape(I[k].shape[0],I[k].shape[1]-2*(i-k),3)
			if i!=0:
				masks[0]=masks[0][masks[0]].reshape(masks[0].shape[0],masks[0].shape[1]-2*(i),3)
			
	# show_img("img",I[0],0)
	return I[0]

def fast_vertical_removal(image,N,depth,energy=3):
	I,s1,s2=odd_to_even_size_1(image)
	I=fast_seam_removal(I,N,depth,energy)
	if  len(s2)!=0:
			return np.hstack((I,np.rollaxis(s2,1)))
	else:
		return I

def fast_horizontal_removal(image,N,depth,seam_r,energy=3):
	I,s1,s2=odd_to_even_size_2(image)
	I=np.rollaxis(I,1)
	I=fast_seam_removal(I,N,depth,energy)
	I=np.rollaxis(I,1)
	if len(s1)!=0:
			return np.vstack((I,s1))
	else:
		return I

def fast_seam_carve_reduction(image,r,c,depth,energy):
	fin_image=fast_vertical_removal(image,c,depth,energy)
	fin_image=fast_horizontal_removal(fin_image,r,depth,energy)
	return fin_image

if __name__=="__main__":
	time1=time()
	image1=cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
	print(image1.shape)
	img1=fast_seam_carve_reduction(image1,60,0,3,2)
	print(img1.shape)
	print(time()-time1)
	# show_img("img1",img1,0)
	cv2.imwrite("fast_seam_50_hor.jpg",img1)


