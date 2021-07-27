import sys, math, cv2
import numpy as np
from collections import defaultdict,OrderedDict

input_path, output_path, colors,dithering= None, None, None, "popularity"
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
		colors=math.log2(int(args[i]))
	if args[i]=="--quantizer":
		i=i+1
		dithering=args[i]
	i=i+1

original_image=cv2.imread(input_path)
print("image shape : ", original_image.shape)


info = []
if dithering=="median_cut":
	for i in range(original_image.shape[0]):
		for j in range(original_image.shape[1]):
			info.append([original_image[i,j][0],original_image[i,j][1],original_image[i,j][2],i,j])

centroids=[]

def median_cut(image, info, colors):
	if colors == 0:
		centroid=np.mean(info,axis=0)
		centroids.append([centroid[0], centroid[1], centroid[2]])
		return

	color_var=np.var(info,axis=0)

	max_variance= np.array([color_var[0],color_var[1],color_var[2]]).argsort()[2]

	info = info[info[:,max_variance].argsort()]
	centre = info.shape[0] // 2

	median_cut(image, info[0:centre,:], colors-1)
	median_cut(image, info[centre:,:], colors-1)

	return 

def popularity(image,colors):
	flat=image.reshape(-1,3).tolist()
	histogram=defaultdict(int)
	for pixel in flat:
		histogram[tuple(pixel)]+=1

	sorted_histogram=OrderedDict(sorted(histogram.items(), key=lambda kv: kv[1],reverse=True))

	k=0
	for key,v in sorted_histogram.items():
		if k<=colors:
			centroids.append([key[0],key[1],key[2]])
			k+=1

	return 


def choose_color(pixel,centroids):
	arr=[]
	for centroid in centroids:
		arr.append(np.sum(np.square(pixel-np.array(centroid))))
	return centroids[np.argsort(arr)[0]]


def quantize_and_dither(image,centroids):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			e=image[i,j].astype(np.uint8)
			image[i,j]=choose_color(image[i,j],centroids)
			if dithering==True:
				e=e-image[i,j]
				if j<image.shape[1]-1:
					image[i,j+1]=image[i,j+1] + (e*7)//16
				if i<image.shape[0]-1 and j>1:
					image[i+1,j-1]=image[i+1,j-1] + (e*3)//16
				if j< image.shape[1]-1 and i <image.shape[0]-1:
					image[i+1,j+1]=image[i+1,j+1] + (e*1)//16
				if i<image.shape[0] -1:
					image[i+1,j]=image[i+1,j] + (e*5)//16
		

	return

if dithering=="median_cut":
	median_cut(original_image, np.array(info), colors)
else:
	popularity(original_image,colors)

quantize_and_dither(original_image,centroids)
cv2.imwrite(output_path,original_image)


