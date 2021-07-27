import cv2
import numpy as np
import sys
from energy import *
from mask import *
from time import time

def show_img(title,img,wait):
	cv2.imshow(title, img)
	k = cv2.waitKey(wait)
	cv2.destroyWindow(title)
	return k

def vertical_seam(energy):
	dp=np.zeros((energy.shape))
	path=np.zeros((energy.shape),dtype=np.int64)
	dp[0,:]=energy[0,:]
	for y in range(1,energy.shape[0]):
		for x in range(energy.shape[1]):
			min_val_x=None
			if x==0:
				min_val_x=np.argmin([dp[y-1,x],dp[y-1,x+1]])
			elif x==energy.shape[1]-1:
				min_val_x=np.argmin([dp[y-1,x-1],dp[y-1,x]]) -1
			else:
				min_val_x=np.argmin([dp[y-1,x-1],dp[y-1,x],dp[y-1,x+1]]) -1

			dp[y,x]=energy[y,x] + dp[y-1,x+min_val_x]
			path[y,x]=min_val_x

	min_seam = np.argmin(dp[energy.shape[0]-1,:])
	min_val=np.min(dp[energy.shape[0]-1,:])
	path_seam=[[energy.shape[0]-1,min_seam]]

	for y in range(energy.shape[0]-2,-1,-1):
		min_seam=min_seam+path[y+1,min_seam]
		path_seam.append([y,min_seam])

	path_seam.reverse()
	return np.array(path_seam,dtype=np.int64),min_val

def horizontal_seam(energy):
	dp=np.zeros((energy.shape))
	path=np.zeros((energy.shape),dtype=np.int64)
	dp[:,0]=energy[:,0]
	for x in range(1,energy.shape[1]):
		for y in range(energy.shape[0]):
			min_val_y=None
			if y==0:
				min_val_y=np.argmin([dp[y,x-1],dp[y+1,x-1]])
			elif y==energy.shape[0]-1:
				min_val_y=np.argmin([dp[y-1,x-1],dp[y,x-1]]) -1
			else:
				min_val_y=np.argmin([dp[y-1,x-1],dp[y,x-1],dp[y+1,x-1]]) -1

			dp[y,x]=energy[y,x] + dp[y+min_val_y,x-1]
			path[y,x]=min_val_y

	min_seam = np.argmin(dp[:,energy.shape[1]-1])
	min_val=np.min(dp[:,energy.shape[1]-1])
	path_seam=[[min_seam,energy.shape[1]-1]]

	for x in range(energy.shape[1]-2,-1,-1):
		min_seam=min_seam+path[min_seam,x+1]
		path_seam.append([min_seam,x])

	path_seam.reverse()
	return np.array(path_seam,dtype=np.int64),min_val



def remove_horizontal_seam(image,pos=0,energy_func=3):
	fin_image=image.copy()
	img_gray,_,_=cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
	e=energy_of_image(img_gray,energy_func)
	seams,min_h=horizontal_seam(e)

	for s in seams:
		fin_image[s[0],s[1]]=[0,0,255]

	# show_img('image',fin_image,0)

	fin_image=np.rollaxis(fin_image,1)
	mask = np.ones(fin_image.shape,dtype=bool)
	for s in seams:
		mask[s[1],s[0]]=0

	cv2.imwrite("opt_red/"+str(pos)+".jpg",np.rollaxis(fin_image,1))
	fin_image= fin_image[mask].reshape(fin_image.shape[0],fin_image.shape[1]-1,3)
	fin_image=np.rollaxis(fin_image,1)


	# print(fin_image.shape)

	return fin_image,seams,min_h

def remove_vertical_seam(image,pos=0,energy_func=3):

	fin_image=image.copy()
	img_gray,_,_=cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
	e=energy_of_image(img_gray,energy_func)
	seams,min_v=vertical_seam(e)

	mask = np.ones(fin_image.shape,dtype=bool)
	for s in seams:
		fin_image[s[0],s[1]]=[0,0,255]
		mask[s[0],s[1]]=0

	# show_img("image",fin_image,0)
	cv2.imwrite("opt_red/"+str(pos)+".jpg",fin_image)

	fin_image= fin_image[mask].reshape(fin_image.shape[0],fin_image.shape[1]-1,3)

	# print(fin_image.shape)

	return fin_image,seams,min_v

def image_resize(image,r,c,energy_func=3):
	fin_image=image.copy()

	if r<0 and c<0:
		return retarget_optimal_seam_order(image,-1*r,-1*c,energy_func)

	else:

		if c<0:
			for i in range(-1*c):
				fin_image,_,_=remove_vertical_seam(fin_image,i,energy_func)
		elif c>0:
			
			fin_image=vertical_seam_insertions(fin_image,c,energy_func)

		if r<0:
			for i in range(-1*r):
				fin_image,_,_=remove_horizontal_seam(fin_image,i,energy_func)
		elif r>0:
			fin_image=horizontal_seam_insertions(fin_image,r,energy_func)

	return fin_image

def xy_diameters(mask):
	x_min,y_min=10000,10000
	x_max,y_max=-1,-1
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i,j]!=0:
				x_min=min(j,x_min)
				x_max=max(j,x_max)
				y_min=min(i,y_min)
				y_max=max(i,y_max)

	return x_max-x_min,y_max-y_min


def object_removal(img,mask):
	img_copy = img.copy()
	i=0
	while(len(mask[mask>0])!= 0):
		print(i)
		img_copy2 = img_copy.copy()
		# img_energy = mask_energy(cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY),mask)
		img_energy=energy_of_image(cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY),4,mask)
		x_dia,y_dia=xy_diameters(mask)
		x_dia,y_dia=1,2
		if x_dia<=y_dia:
			seam_mask = np.ones(img_copy.shape,dtype=bool)
			seams,_ = vertical_seam(img_energy)
			for s in seams:
				seam_mask[s[0],s[1]]=0
			for s in seams:
				img_copy2[s[0],s[1]]=[0,0,255]
			# show_img("img",img_copy2,0)
			print(img_copy2.shape)
			cv2.imwrite("obj_rem_vert_multi/"+str(i)+".jpg",img_copy2)
			img_copy = img_copy[seam_mask].reshape(img_copy.shape[0],img_copy.shape[1]-1,3)
			mask = mask[seam_mask[:,:,0]].reshape(img_copy.shape[0],img_copy.shape[1])
		elif x_dia>y_dia:
			img_copy = np.rollaxis(img_copy,1)
			mask = np.rollaxis(mask,1)
			seam_mask = np.ones(img_copy.shape,dtype=bool)
			seams,_ = horizontal_seam(img_energy)
			for s in seams:
				seam_mask[s[1],s[0]]=0
			for s in seams:
				img_copy2[s[0],s[1]]=[0,0,255]
			# show_img("img",img_copy2,0)
			cv2.imwrite("obj_rem_hor_multi/"+str(i)+".jpg",img_copy2)
			img_copy = img_copy[seam_mask].reshape(img_copy.shape[0],img_copy.shape[1]-1,3)
			mask = mask[seam_mask[:,:,0]].reshape(img_copy.shape[0],img_copy.shape[1])
			img_copy = np.rollaxis(img_copy,1)
			mask = np.rollaxis(mask,1)
			
		i=i+1

	delta_x=img.shape[1]-img_copy.shape[1]
	delta_y=img.shape[0]-img_copy.shape[0]
	print(img_copy.shape)
	img_copy=vertical_seam_insertions(img_copy,delta_x)
	img_copy=horizontal_seam_insertions(img_copy,delta_y)
	print(img_copy.shape)
	return img_copy



def add_single_seam(image,seam,hor,pos):
	fin_image = np.zeros((image.shape[0], image.shape[1] + 1, 3))
	print(pos)
	for s in seam:
		r,c=s[0],s[1]
		if c!=0:
			avg=np.mean(image[r,c-1:c+1,:],axis=0)
			fin_image[r,:c]=image[r,:c]
			fin_image[r,c]=avg
			fin_image[r,c+1:]=image[r,c:]
		else:
			avg=np.mean(image[r,c:c+1,:],axis=0)
			fin_image[r,c]=image[r,c]
			fin_image[r,c+1]=avg
			fin_image[r,c+1:]=image[r,c:]
	img_copy=fin_image.copy().astype('uint8')
	for s in  seam:
		img_copy[s[0],s[1]]=[0,0,255]
	if hor:
		# show_img("seam",np.rollaxis(img_copy,1),0)
		cv2.imwrite("hor_ins_rem/"+str(pos)+".jpg",np.rollaxis(img_copy,1))
	else:
		# sho
		print(pos)
		cv2.imwrite("vert_ins_rem/"+str(pos)+".jpg",img_copy)
	return fin_image.astype('uint8')
			
		
def vertical_seam_insertions(image, n,energy_func=3,hor=False):
	seams = []
	fin_img,temp_img = image.copy(),image.copy()
  
	for i in range(n):
		temp_img,seam,_ = remove_vertical_seam(temp_img,energy_func=energy_func)
		print(i)
		seams.append(seam)

	seams.reverse()

	for i in range(n):
		seam = seams.pop()
		fin_img = add_single_seam(fin_img, seam,hor,i)

		for remaining_seam in seams:  
			for j in range(len(remaining_seam)):
				if  remaining_seam[j][1]>=seam[j][1]:
					remaining_seam[j][1]+=2

	return fin_img

def horizontal_seam_insertions(image,n,energy_func=3):
	fin_img=np.rollaxis(image,1)
	fin_img=vertical_seam_insertions(fin_img,n,energy_func,hor=True)
	return np.rollaxis(fin_img,1)


def retarget_optimal_seam_order(img,r,c,energy_func=3):
	fin_image=img.copy()
	i=0
	while True:
		print(i)
		if r==0 and c!=0:
			fin_image,_,_=remove_vertical_seam(fin_image,i,energy_func)
			c=c-1
		elif c==0 and r!=0:
			fin_image,_,_=remove_horizontal_seam(fin_image,i,energy_func)
			r=r-1

		elif c!=0 and r!=0:
			_,ev=vertical_seam(energy_of_image(cv2.cvtColor(fin_image.copy(), cv2.COLOR_BGR2GRAY),energy_func))
			_,eh=horizontal_seam(energy_of_image(cv2.cvtColor(fin_image.copy(), cv2.COLOR_BGR2GRAY),energy_func))
			if ev<=eh:
				fin_image,_,_=remove_vertical_seam(fin_image,i,energy_func)
				c=c-1
			else:
				fin_image,_,_=remove_horizontal_seam(fin_image,i,energy_func)
				r=r-1
		elif r==0 and c==0:
			break
		i=i+1


	return fin_image.astype('uint8')


if __name__=="__main__":
	image1=cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
	print(image1.shape)
	# img1=image_resize(image1,-2,3,1)
	# mask=make_mask(image1,int(sys.argv[2]))
	# show_img("mask",mask[:,:,0],0)

	# img1=object_removal(image1,mask[:,:,0])
	# img1=image_resize(image1,-25,-25)
	# img1=retarget_image(image1,3,3)
	# img1=retarget_optimal_seam_order(image1,5,5)
	# print(img1.shape)
	# show_img("img1",img1,0)
	# cv2.imwrite("mask_vert.jpg",mask)
	# cv2.imwrite("obj_vert_rem.jpg",img1)


	# canvas=np.zeros((image1.shape)).astype('uint8')
	# path="opt_red/"
	# path2="opt_red/same_size/"
	# for i in range(50):
	# 	img=cv2.imread(path+str(i)+".jpg",cv2.IMREAD_COLOR)
	# 	paste=canvas.copy()
	# 	paste[:img.shape[0],:img.shape[1],:]=img
	# 	cv2.imwrite(path2+str(i)+".jpg",paste)

	canvas=np.zeros((image1.shape)).astype('uint8')
	path="vert_ins_rem/"
	path2="vert_ins_rem/same_size/"
	for i in range(121):
		img=cv2.imread(path+str(i)+".jpg",cv2.IMREAD_COLOR)
		paste=canvas.copy()
		paste[:img.shape[0],:img.shape[1],:]=img
		cv2.imwrite(path2+str(121+i)+".jpg",paste)

	# path1="hor_ins_rem/same_size/"
	# path2="obj_rem_2/"
	# for i in range(43):
	# 	img=cv2.imread(path1+str(i)+".jpg",cv2.IMREAD_COLOR)
	# 	cv2.imwrite(path2+str(43+i)+".jpg",img)








