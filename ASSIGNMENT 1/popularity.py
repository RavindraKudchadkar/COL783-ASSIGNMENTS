import sys, math, cv2
import numpy as np
from collections import defaultdict,OrderedDict

input_path, output_path, colors,dithering= None, None, None, False
args=sys.argv
del args[0]

i=0
while (i<len(args)):
	if args[i]=="--input":
		i=i+1
		input_path=args[i]
	if args[i]=="--output":
		i=i+1
		output_path=args[i]
	if args[i]=="--colors":
		i=i+1
		colors=int(args[i])
	if args[i]=="dither":
		dithering=True
	i=i+1

original_image=cv2.imread(input_path)
print("image shape : ",original_image.shape)

def popularity(image,colors):
	flat=image.reshape(-1,3).tolist()
	histogram=defaultdict(int)
	for pixel in flat:
		histogram[tuple(pixel)]+=1

	sorted_histogram=OrderedDict(sorted(histogram.items(), key=lambda kv: kv[1],reverse=True))

	k=0
	quantizers=[]
	for key,v in sorted_histogram.items():
		if k<=colors:
			quantizers.append([key[0],key[1],key[2]])
			k+=1

	return np.array(quantizers)

def choose_color(pixel,centroids):
	arr=[]
	for centroid in centroids:
		arr.append(np.sum(np.square(pixel-centroid)))
	return centroids[np.argsort(arr)[0]]

def quantize(image,quantizers):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			e=image[i,j].astype(np.uint8)
			image[i,j]=choose_color(image[i,j],quantizers)
			# if dithering==True:
			# 	e=e-image[i,j]
			# 	if i<image.shape[0]-1:
			# 		image[i+1,j]=image[i+1,j] + (e*7)//16
			# 	if i>1 and j< image.shape[1]-1:
			# 		image[i-1,j+1]=image[i-1,j+1] + (e*3)//16
			# 	if j< image.shape[1]-1 and i <image.shape[0]-1:
			# 		image[i+1,j+1]=image[i+1,j+1] + (e*1)//16
			# 	if j<image.shape[1] -1:
			# 		image[i,j+1]=image[i,j+1] + (e*5)//16


quantizers=popularity(original_image,colors)

quantize(original_image,quantizers)
cv2.imwrite(output_path,original_image)

