import sys
import cv2
import numpy as np
import random
import math
np.seterr(over='ignore')


def luminenceRemapping (source, target):
	source_std = math.sqrt(np.var(source))
	target_std = math.sqrt(np.var(target))
	source_mean = np.mean(source)
	target_mean = np.mean(target)
	source = target_std/source_std*(source - source_mean) + target_mean
	source_min = np.min(source)
	source = source - source_min
	source_max = np.max(source)
	source = source.astype(float)
	source = source * 255
	source = source / source_max
	source = np.floor(source)
	source = source.astype('uint8')
	return source


def showImage(img):
	cv2.imshow("Sample", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def jittered(l, a, b, n, sam):
	length_x = len(l[0])
	length_y = len(l)
	partition_x = length_x/n
	partition_y = length_y/n
	for i in range(0, n):
		for j in range(0, n):
			initial_x = int(i * partition_x)
			initial_y = int(j * partition_y)
			final_x = initial_x + 1
			final_y = initial_y + 1
			if(n == n-1):
				final_x = length_x-1
				final_y = length_y-1
			final_x = int(final_x + partition_x)
			final_y = int(final_y + partition_y)
			range_x = final_x - initial_x
			range_y = final_y - initial_y
			x = int(initial_x + random.random() * range_x)
			y = int(initial_y + random.random() * range_y)
			value_l = l[y][x]
			if value_l not in sam:
				sam[value_l] = [a[y][x], b[y][x]]
	return sam


def colourise(tar_l):
	# ith row, jth col
	either = window_size / 2
	either = math.floor(either)
	either = int(either)
	global target_a
	global target_b
	global samplings
	global swatches
	for i in range(len(tar_l)):
		for j in range(len(tar_l[0])):
			
			lum_tar = tar_l[i][j]
			errors = {}
			for key in samplings:
				er = 0
				temp_sam = samplings[key]
				temp_sw = swatches[key]
				top_left = temp_sw[2]
				bottom_right = temp_sw[3]
				swa = tar_l[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
				swa = swa.astype('double')
				elems = len(swa) * len(swa[0])
				# print(swa)
				for g in range(i-either, i+either+1):
					for h in range(j-either, j+either+1):
						if g < 0 or g > len(tar_l)-1 or h < 0 or h > len(tar_l[0])-1:
							continue
						# print(swa)
						swat = swa.copy()
						swat = swat - tar_l[g][h]
						swat = np.square(swat)
						swat = swat.sum()
						swat = swat/elems
						er = er + swat


				errors[key] = er 
				# print(er)
			min_er = min(errors.values())
			key = [key for key in errors if errors[key] == min_er]
			key = key[0]
			sam = samplings[key]

			pair_ab = sam.get(lum_tar) or sam[min(sam.keys(), key = lambda key: abs(key - lum_tar))]
			# res = test_dict.get(search_key) or test_dict[min(test_dict.keys(), key = lambda key: abs(key-search_key))] 
			target_a[i][j] = pair_ab[0]
			target_b[i][j] = pair_ab[1]
	merged = cv2.merge([tar_l,target_a,target_b])
	return merged


def colourise_swatch(sam, tar_l, points):
	# ith row, jth col
	global target_a
	global target_b
	top_left = points[2]
	bottom_right = points[3]
	for i in range(top_left[1], bottom_right[1]):
		for j in range(top_left[0], bottom_right[0]):
			if not (target_a[i][j] == 129):
				continue
			lum_tar = tar_l[i][j]
			pair_ab = sam.get(lum_tar) or sam[min(sam.keys(), key = lambda key: abs(key - lum_tar))]
			# res = test_dict.get(search_key) or test_dict[min(test_dict.keys(), key = lambda key: abs(key-search_key))] 
			target_a[i][j] = pair_ab[0]
			target_b[i][j] = pair_ab[1]


def swatch_select(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		swatch_points.append((x,y))
		selected = True
	elif event == cv2.EVENT_LBUTTONUP:
		swatch_points.append((x,y))
		selected = False


source_path, target_path, output_path, window_size= None, None, None, 0
args=sys.argv
del args[0]

i=0
while (i<len(args)):
	if args[i]=="--source":
		i=i+1
		source_path=args[i]
	if args[i]=="--output":
		i=i+1
		output_path=args[i]
	if args[i]=="--target":
		i=i+1
		target_path=args[i]
	if args[i]=="--window_size":
		i=i+1
		window_size = int(args[i])
	i=i+1

coloured = cv2.imread(source_path, cv2.IMREAD_COLOR)
coloured_lab = cv2.cvtColor(coloured, cv2.COLOR_BGR2LAB)
target = cv2.imread(target_path, 0)

sw = 'y'
samplings = {}
swatches = {}
i = 0
target_a = np.zeros_like(target)
target_b = np.zeros_like(target)
target_a = target_a -127
target_b = target_b -127
while sw == 'y':
	swatch_points = []
	selected = False
	i += 1

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", swatch_select)
	cv2.imshow("image", coloured)
	key = cv2.waitKey(0)
	coloured_roi = coloured_lab[swatch_points[0][1]:swatch_points[1][1], swatch_points[0][0]:swatch_points[1][0]]
	
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", swatch_select)
	cv2.imshow("image", target)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	target_roi = target[swatch_points[2][1]:swatch_points[3][1], swatch_points[2][0]:swatch_points[3][0]]
	swatches[i] = swatch_points

	coloured_roi_copy = coloured_roi.copy()
	coloured_roi_l, coloured_roi_a, coloured_roi_b = cv2.split(coloured_roi_copy)
	coloured_roi_l = luminenceRemapping(coloured_roi_l,target_roi)
	samplings_temp = {}
	samplings_temp = jittered(coloured_roi_l, coloured_roi_a, coloured_roi_b, 25, samplings_temp)
	# samplings[i] = samplings_temp
	colourise_swatch(samplings_temp, target, swatch_points)

	target_roi_a = target_a[swatch_points[2][1]:swatch_points[3][1], swatch_points[2][0]:swatch_points[3][0]]
	target_roi_b = target_b[swatch_points[2][1]:swatch_points[3][1], swatch_points[2][0]:swatch_points[3][0]]
	samplings_temp = jittered(target_roi, target_roi_a, target_roi_b, 25, samplings_temp)
	samplings[i] = samplings_temp


	print("Another Swatch?")
	sw = input()
# print(target_a)
# print(target_b)
merged = cv2.merge([target, target_a, target_b])
merged = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
showImage(merged)
# print(samplings)
merged = colourise(target)
merged = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
# showImage(merged)
cv2.imwrite(output_path, merged)