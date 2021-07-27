import sys, math, cv2
import numpy as np


input_path, output_path, colors= None, None, None
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
	if args[i]=="dither":
		dithering=True
	i=i+1

quantized_image=cv2.imread(input_path)
print("image shape : ",quantized_image.shape)


info = []
for i in range(quantized_image.shape[0]):
	for j in range(quantized_image.shape[1]):
		info.append([quantized_image[i,j][0],quantized_image[i,j][1],quantized_image[i,j][2],i,j])


def median_cut(image, info, colors):
	if colors == 0:
		centroid=np.mean(info,axis=0)
		for item in info:
			quantized_image[item[3],item[4]] = [centroid[0], centroid[1], centroid[2]]
		return

	color_var=np.var(info,axis=0)
	# color_var=np.max(info,axis=0)-np.min(info,axis=0)

	max_variance= np.array([color_var[0],color_var[1],color_var[2]]).argsort()[2]

	info = info[info[:,max_variance].argsort()]
	centre = info.shape[0] // 2

	median_cut(image, info[0:centre,:], colors-1)
	median_cut(image, info[centre:,:], colors-1)

	return 

median_cut(quantized_image, np.array(info), colors)
cv2.imwrite(output_path,quantized_image)


