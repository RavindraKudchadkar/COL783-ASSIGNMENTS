import sys
import numpy as np
import cv2
import random
import math
np.seterr(over='ignore')

def luminenceRemapping (source, target):
	# print(source)
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
	# print(source)
	return source


def showImage(img):
	cv2.imshow("Sample", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def jittered(l, a, b, n):
	sam = {}
	length_x = len(l[0])
	length_y = len(l)
	partition_x = length_x/n
	partition_y = length_y/n
	for i in range(0, n-1):
		for j in range(0, n-1):
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


def colourise(sam, tar_l):
	# ith row, jth col
	tar_a = np.zeros_like(tar_l)
	tar_b = np.zeros_like(tar_l)
	for i in range(len(tar_l)):
		for j in range(len(tar_l[0])):
			lum_tar = tar_l[i][j]
			pair_ab = sam.get(lum_tar) or sam[min(sam.keys(), key = lambda key: abs(key - lum_tar))]
			# res = test_dict.get(search_key) or test_dict[min(test_dict.keys(), key = lambda key: abs(key-search_key))] 
			tar_a[i][j] = pair_ab[0]
			tar_b[i][j] = pair_ab[1]
	merged = cv2.merge([tar_l,tar_a,tar_b])
	return merged


source_path, target_path, output_path= None, None, None
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
	i=i+1

coloured = cv2.imread(source_path, cv2.IMREAD_COLOR)
coloured_lab = cv2.cvtColor(coloured, cv2.COLOR_BGR2LAB)
coloured_l, coloured_a, coloured_b = cv2.split(coloured_lab)
target = cv2.imread(target_path, 0)


# print(target)
coloured_l = luminenceRemapping(coloured_l, target)
n = 16
samples = jittered(coloured_l, coloured_a, coloured_b, n)
# samples = sorted(samples)
# print(samples)
colourised_target = colourise(samples, target)
# showImage(colourised_target)
colourised_target = cv2.cvtColor(colourised_target, cv2.COLOR_LAB2BGR)
cv2.imwrite(output_path, colourised_target)