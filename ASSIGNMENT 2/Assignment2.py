import sys
import numpy as np
import cv2
import math
import scipy
import scipy.sparse
from scipy.sparse.linalg import spsolve
import imutils
from imutils import face_utils
np.seterr(over='ignore')


source = cv2.imread("reference.jpg", cv2.IMREAD_COLOR)
target = cv2.imread("source.jpg", cv2.IMREAD_COLOR)
target_dim = (len(target[0]), len(target))
source = cv2.resize(source, target_dim)


# POINTS OF INTEREST

class forehead:
    def __init__(self,feature_points,image):
        print("Select the 91 points")
        cv2.namedWindow('Select_Points')
        cv2.setMouseCallback('Select_Points', self.select_point)
        self.points = np.zeros((91, 2), dtype="int")
        self.count = 0
        self.inputFeature = 0
        self.output_marked_image = np.copy(image)
        for (x, y) in feature_points:
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
        while self.count != 91:
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
        if self.count == 91:
            print("Points successfully selected")

def feature_points_detection(img):
    
    img_color = img.copy()
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)


    points = []

    fore = forehead(points, img_color)
    points = fore.points
    points = points.tolist()
    cv2.imshow("dots", fore.getOutputMarkedImage())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    x = img.shape[1]
    y = img.shape[0]
    lx = 0
    ly = 0
    for p in points:
        px = p[0]
        py = p[1]
        if px < x:
            x = px
        if py < y:
            y = py
        if px > lx:
            lx = px
        if py > ly:
            ly = py
    rec = (x, y, lx-x+1, ly-y+1)
    # print(rec)
    return points, rec





source_points, src_rectangle = feature_points_detection(source)





target_points, dst_rectangle = feature_points_detection(target)


# TRIANGLES




def triangulate(points, rec, shape):
    sub = cv2.Subdiv2D(rec)
    for p in points:
        # print(p)
        sub.insert((p[0],p[1]))
    tria_list = sub.getTriangleList()
    # print(tria_list)
    # tria_list = tria_list.astype(int)
    tri_list = []
    for t in tria_list:
        if t[0] < rec[0] or t[1] < rec[1] or t[2] < rec[0] or t[3] < rec[1] or t[4] < rec[0] or t[5] < rec[1] or t[0] > rec[0]+rec[2] or t[1] > rec[1]+rec[3] or t[2] > rec[0]+rec[2] or t[3] > rec[1]+rec[3] or t[4] > rec[0]+rec[2] or t[5] > rec[1]+rec[3]:
            continue
        else:
            tri_list.append(t.tolist())
    # print(tri_list)
    tri_list = np.array(tri_list).astype('uint')
    tri_list = tri_list.tolist()
    return tri_list





src_triangles = triangulate(source_points, src_rectangle, source.shape)
dst_triangles = triangulate(target_points, dst_rectangle, target.shape)


# WARP




def area(x1,y1,x2,y2,x3,y3):
    return math.fabs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2.0)

def PointInsideTriangle(point,polygon):
    hull = cv2.convexHull(np.array(polygon))
    dist = cv2.pointPolygonTest(hull,(point[0], point[1]),False)
    if dist>=0:
        return True
    else:
        return False

def warp(src, dst, src_points, dst_points, src_triangles, dst_triangles):
    warped_image = np.zeros(src.shape, dtype=np.uint8)
#     print(src.shape)
#     print(dst.shape)
    for t in dst_triangles:
        first, second, third = -1,-1,-1
        for i2, (x,y) in enumerate(dst_points):
            if(x==t[0] and y==t[1]):
                first = i2
            if(x==t[2] and y==t[3]):
                second = i2
            if(x==t[4] and y==t[5]):
                third = i2
        if(first>=0 and second>=0 and third>=0):
            x1,y1 = src_points[first]
            x2,y2 = src_points[second]
            x3,y3 = src_points[third]
            dx1,dy1 = dst_points[first]
            dx2,dy2 = dst_points[second]
            dx3,dy3 = dst_points[third]
            # print(x1,y1,x2,y2,x3,y3,dx1,dy1,dx2,dy2,dx3,dy3)
            triangle_area=area(dx1,dy1,dx2,dy2,dx3,dy3)
            warped_image[dy1,dx1]=src[y1,x1]
            warped_image[dy2,dx2]=src[y2,x2]
            warped_image[dy3,dx3]=src[y3,x3]
            min_x,max_x=min(dx1,dx2,dx3),max(dx1,dx2,dx3)
            min_y,max_y=min(dy1,dy2,dy3),max(dy1,dy2,dy3)

            for i in range(min_x,max_x+1,1):
                for j in range(min_y,max_y+1,1):
                    if PointInsideTriangle((j,i),np.array([[dy1,dx1],[dy2,dx2],[dy3,dx3]])):

                        b1=area(i,j,dx3,dy3,dx2,dy2)/triangle_area
                        b2=area(i,j,dx3,dy3,dx1,dy1)/triangle_area
                        b3=area(i,j,dx1,dy1,dx2,dy2)/triangle_area
                        warped_image[j,i]=src[int(b1*y1+b2*y2+b3*y3), int(b1*x1+b2*x2+b3*x3)]

    return warped_image





warped_src = warp(source, target, source_points, target_points, src_triangles, dst_triangles)
cv2.imshow("warped", warped_src)
cv2.waitKey(0)
cv2.destroyAllWindows()





cv2.imwrite("warp.jpg", warped_src)





def delaunay(img, triangleList, output):
    color = (255,0,0)
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA, 0)
    cv2.imwrite(output, img)

delaunay(source, src_triangles, "delaunay_src.jpg")
delaunay(target, dst_triangles, "delaunay_dst.jpg")


# DECOMPOSITION




def skin_details(src, dia, sig_s, sig_v, cmat ):
    return src - filter_bilateral(src.astype('float32'), dia, sig_s, sig_v, cmat).astype('uint8')


def gaussian(r2, sigma):
    return (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0


def filter_bilateral( img_in, diameter, sigma_s, sigma_v, cmat, reg_constant=1e-8):

    window_size = diameter//2

    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    for s_x in range(-window_size,window_size+1):
        for s_y in range(-window_size,window_size+1):
            w = gaussian( s_x**2+s_y**2, sigma_s )
            offset = np.roll(img_in, [s_y, s_x], axis=[0,1] )
            tw = w*gaussian( (offset-img_in)**2, sigma_v )

            result += offset*tw
            wgt_sum += tw

    result = result/wgt_sum
    result = result.astype('uint8')
    out = np.zeros_like(img_in)

    for i in range(len(img_in)):
        for j in range(len(img_in[0])):
            if cmat[i][j] == 2 or cmat[i][j] == 3 or cmat[i][j] == 0:
                out[i][j] = img_in[i][j]
            else:
                out[i][j] = result[i][j]
    return out.astype('uint8')





def classes(points,shape):
    cmat = np.zeros((shape[0],shape[1]))

    left_eye = points[53:61] 
    right_eye = points[61:69]
    left_eyebrow1 = points[31:36]
    left_eyebrow2 = points[30:37]
    right_eyebrow1 = points[37:39] + points[41:44]
    right_eyebrow2 = points[38:42]
    mouth = points[83:89]
    lips1 = points[69:72] + points[83:85] + points[80:81] 
    lips2 = points[71:74] + points[84:86] + points[81:82]
    lips3 = points[74:80] + points[86:89] + points[82:83]
    lips4 = points[79:81] + points[83:84] + points[88:89]
    lips5 = points[81:83] + points[85:87]
    skin = points[0:30]

    for y in range(shape[0]):
        for x in range(shape[1]):
            if PointInsideTriangle((x,y),skin):
                if PointInsideTriangle((x,y),left_eye):
                    cmat[y][x] = 3
                elif PointInsideTriangle((x,y),right_eye):
                    cmat[y][x] = 3
                elif PointInsideTriangle((x,y), left_eyebrow1) or PointInsideTriangle((x,y), left_eyebrow2):
                    cmat[y][x] = 4
                elif PointInsideTriangle((x,y), right_eyebrow1) or PointInsideTriangle((x,y), right_eyebrow2):
                    cmat[y][x] = 4
                elif PointInsideTriangle((x,y),lips1) or PointInsideTriangle((x,y),lips2) or PointInsideTriangle((x,y),lips3) or PointInsideTriangle((x,y),lips4) or PointInsideTriangle((x,y),lips5):
                    cmat[y][x] = 2
                elif PointInsideTriangle((x,y),mouth):
                    cmat[y][x] = 3                        
                else:
                    cmat[y][x] = 1

    return cmat


def beta_function(points,shape, mask):
    beta = np.zeros((shape[0],shape[1])).astype('float32')
    sigma_sq = min(shape[0], shape[1])


    left_eye = points[53:61] 
    right_eye = points[61:69]
    left_eyebrow1 = points[31:36]
    left_eyebrow2 = points[30:32] + points[35:37]
    right_eyebrow1 = points[37:39] + points[41:44]
    right_eyebrow2 = points[38:42]
    mouth = points[83:89]
    lips1 = points[69:72] + points[83:85] + points[80:81] 
    lips2 = points[71:74] + points[84:86] + points[81:82] 
    lips3 = points[74:80] + points[86:89] + points[82:83]
    lips4 = points[79:81] + points[83:84] + points[88:89]
    lips5 = points[81:83] + points[85:87]
    skin = points[0:30]
    
    nose = points[44:53]
    nostril = points[89:91]
    
    for y in range(shape[0]):
        for x in range(shape[1]):
            if PointInsideTriangle((x,y),skin):
                if PointInsideTriangle((x,y),left_eye):
                    beta[y][x] = 0
                elif PointInsideTriangle((x,y),right_eye):
                    beta[y][x] = 0
                elif PointInsideTriangle((x,y), left_eyebrow1) or PointInsideTriangle((x,y), left_eyebrow2):
                    beta[y][x] = 0.3
                elif PointInsideTriangle((x,y), right_eyebrow1) or PointInsideTriangle((x,y), right_eyebrow2):
                    beta[y][x] = 0.3
                elif PointInsideTriangle((x,y),lips1) or PointInsideTriangle((x,y),lips2) or PointInsideTriangle((x,y),lips3) or PointInsideTriangle((x,y),lips4) or PointInsideTriangle((x,y),lips5):
                    beta[y][x] = 0
                else:
                    beta[y][x] = 1
    
    beta = beta*255
    u = 0
    while u < len(nose)-1:
        start = (nose[u][0], nose[u][1])
        end = (nose[u+1][0], nose[u+1][1])
        beta = cv2.line(beta, start, end, 0.05, 1)
        u += 1
    for poi in nostril:
        poi_x = poi[0]
        poi_y = poi[1]
        beta[poi_y, poi_x] = 0
    
    beta = beta/255
    beta_smoothed = np.zeros_like(beta)
    sigma_sq = min(shape[0], shape[1]) / 25
    for y in range(shape[0]): # for p
        for x in range(shape[1]):
            if mask[y][x] == 255:
                beta_p = 1
                for w in range(y-10, y+11): # for q
                    for z in range(x-10, x+11):
                        if w not in range(shape[0]) or z not in range(shape[1]):
                            continue
                        if w == y and z == x:
                            continue
                        if mask[w][z] == 255:
                            k_q = 1-beta[w][z]
                            distance = (y-w)*(y-w) + (x-z)*(x-z)
                            power_of_e = distance / 2
                            power_of_e = (-1) * power_of_e / sigma_sq
                            value_q = 1 - k_q * math.exp(power_of_e)
                            if value_q < beta_p:
                                beta_p = value_q
                beta_smoothed[y][x] = beta_p
#                 print((y,x))
                            
    
    return beta_smoothed


# MASK




warped_l, warped_a, warped_b = cv2.split(cv2.cvtColor(warped_src, cv2.COLOR_BGR2LAB))
target_l, target_a, target_b = cv2.split(cv2.cvtColor(target, cv2.COLOR_BGR2LAB))





vertices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

for i in range(len(vertices)):
    vertices[i] = target_points[vertices[i]]

mask = np.zeros_like(target_l).astype('uint8')
warped_over_target = np.zeros_like(target).astype('uint8')
points_mask = []
cv2.fillPoly(mask, np.array([vertices], dtype = np.int32), 255)
for i in range(len(mask)):
    for j in range(len(mask[0])):
        if mask[i][j] == 255:
            points_mask.append([i,j])
        # if cmat[i][j] == 2 or cmat[i][j] == 3 or cmat[i][j] == 4:
        # 	mask[i][j] = 0
        if warped_src[i][j][0] == 0 and warped_src[i][j][1] == 0 and warped_src[i][j][2] == 0:
            warped_over_target[i][j] = target[i][j]
        elif mask[i][j] == 255:
            warped_over_target[i][j] = warped_src[i][j]
        else:
            warped_over_target[i][j] = target[i][j]
# cv2.imshow("warped over target", warped_over_target)
# cv2.imshow("mask", mask)
# cv2.imshow("skin_details", final_skin_details)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





beta_func_tgt = beta_function(target_points, target_l.shape, mask)





cmat = classes(target_points, target_l.shape)





warp_skin_details = skin_details(warped_l, 5, 15, 15, cmat)
tgt_skin_details = skin_details(target_l, 5, 15, 15, cmat)
warp_smooth = filter_bilateral(warped_l, 5, 15, 15, cmat).astype('uint8')
tgt_smooth = filter_bilateral(target_l, 5, 15, 15, cmat).astype('uint8')


# SKIN AND COLOR TRANSFER




def skin_transfer(warp, tgt, d_w, d_t):

    out =  d_w*warp + d_t*tgt
    return out.astype('uint8')
def colour_transfer_2(src,dst,alpha,cmat):
    out = np.zeros_like(src)
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            if cmat[i,j]==3 or cmat[i][j]==0:
                out[i,j]=dst[i,j]
            elif cmat[i][j] == 2:
                out[i][j] = src[i][j]
            else:
                out[i,j]=(1-alpha)*dst[i,j] + alpha*src[i,j]

    return out.astype("uint8")





final_skin_details = skin_transfer(warp_skin_details, tgt_skin_details, 1, 0)
final_a = colour_transfer_2(warped_a, target_a, 0.7, cmat)
final_b = colour_transfer_2(warped_b, target_b, 0.7, cmat)





final_l = tgt_smooth + final_skin_details





merged = cv2.merge((final_l, final_a, final_b))
merged = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
cv2.imwrite("output.jpg", merged)


# SHADING TRANSFER




from scipy.sparse.linalg import cg

def grad_conv(mat, direction, mask):
    
    result = np.zeros_like(mask).astype('float')
    for y in range(len(mask)):
        for x in range(len(mask[0])):
            if mask[y, x] == 255:
                if y not in range(mask.shape[0]) or x not in range(mask.shape[1]):
                    result[y, x] = float(mat[y,x])
                    continue
                if direction == "x":
                    result[y, x] = float(mat[y,x]) - float(mat[y,x-1])
                elif direction == "y":
                    result[y, x] = float(mat[y,x]) - float(mat[y-1,x]) 
                    
    return result
                
    
    

def shading_transfer(Is, Es, beta, mask):

    size=Is.shape
    
    delta_Rs_x = np.zeros_like(Is).astype('float')
    delta_Rs_y = np.zeros_like(Is).astype('float')
    delta_Is = np.sqrt(np.square(grad_conv(Is, "x", mask)) + np.square(grad_conv(Is, "y", mask)))
    delta_Es = np.sqrt(np.square(grad_conv(Es, "x", mask)) + np.square(grad_conv(Es, "y", mask)))
    delta_Is_x = grad_conv(Is, "x", mask)
    delta_Is_y = grad_conv(Is, "y", mask)
    delta_Es_x = grad_conv(Es, "x", mask)
    delta_Es_y = grad_conv(Es, "y", mask)
    Is_guess = []
    
    total_points = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if mask[i][j] == 255:   
                if beta[i][j] == 0:
                    continue
                if  beta[i][j] * abs(delta_Es[i][j]) > abs(delta_Is[i][j]):
                    delta_Rs_x[i][j] = delta_Es_x[i][j]
                    delta_Rs_y[i][j] = delta_Es_y[i][j]
                else:
                    delta_Rs_x[i][j] = delta_Is_x[i][j]
                    delta_Rs_y[i][j] = delta_Is_y[i][j]
                
                total_points += 1
    
    lap_Rs1 = grad_conv(delta_Rs_x, "x", mask) + grad_conv(delta_Rs_y, "y", mask)
    lap_Rs = []
    for i in range(size[0]):
        for j in range(size[1]):
            if mask[i][j] == 255:
                if beta[i][j] != 0:
                    lap_Rs.append(lap_Rs1[i][j])
                    Is_guess.append(Is[i][j])
    
    Rs = poisson_edit(Is, mask, lap_Rs, beta, Is_guess)
    

    return Rs.astype('uint8')


def look_up_table(mask, beta, tot):

    lut = np.zeros_like(mask).astype('int')
    counter = 0
    for y in range(len(mask)):
        for x in range(len(mask[0])):
            if mask[y, x] == 255 and beta[y,x] != 0:
                counter += 1
                if counter == tot:
                    break
                lut[y, x] = counter

    return lut, counter


def poisson_edit(target, mask, lap_Rs, beta, Is_guess):

    lut, num_points = look_up_table(mask, beta, len(lap_Rs))
    mat_b = np.zeros_like(lap_Rs)
    mat_A = scipy.sparse.lil_matrix((mat_b.shape[0], mat_b.shape[0]))
    
    ind = 0
    for y in range(len(target)):
        for x in range(len(target[0])):
            if mask[y, x] == 255 and beta[y][x] != 0:
                ind += 1
                edge = False
                if ind == mat_b.shape[0]:
                    break
                mat_A[ind, ind] = 4
                                 
                if mask [y, x+1] == 0:
                    mat_b[ind] = target[y, x+1]
                    edge = True
                else: 
                    mat_A[ind,lut[y, x+1]] = -1
                    
                if mask [y+1, x] == 0:
                    mat_b[ind] = target[y+1, x]
                    edge = True
                else: 
                    mat_A[ind,lut[y+1, x]] = -1
                
                if mask [y, x-1] == 0:
                    mat_b[ind] = target[y, x-1]
                    edge = True
                else:
                    mat_A[ind,lut[y, x-1]] = -1
                    
                if mask [y-1, x] == 0:
                    mat_b[ind] = target[y-1, x]
                    edge = True
                else:
                    mat_A[ind,lut[y-1, x]] = -1
                
                mat_b[ind] = mat_b[ind] - lap_Rs[ind]

    mat_A = mat_A.tocsc()

    x1 = cg(mat_A, mat_b)
    x1 = np.array(x1[0])
    
    x1[x1 > 255] = 255
    x1[x1 < 0] = 0
#     x1 = cv2.normalize(x1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    var_src = np.sqrt(np.var(Is_guess))
    mean_src = np.mean(Is_guess)
    mean_x1 = np.mean(x1)
    var_x1 = np.sqrt(np.var(x1))
                      
    for i in range(len(x1)):
        x1[i] = (var_src * (x1[i]-mean_x1) / var_x1) + mean_src
        
    x1[x1 > 255] = 255
    x1[x1 < 0] = 0
                 
    ind1 = 0
#     cv2.imshow("original", target)
    for y in range(len(target)):
        for x in range(len(target[0])):
            if mask[y][x] == 255 and beta[y][x] != 0:
                target[y][x] = x1[ind1]
                ind1 += 1
                if ind1 == len(x1)-1:
                    break
    
#     cv2.imshow("remapped", target)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return target

rs = shading_transfer(tgt_smooth.copy(), warp_smooth, beta_func_tgt, mask)





face_bound = np.zeros_like(mask)
bound_points = target_points[0:30]
u = 0
while u < len(bound_points)-1:
        start = (bound_points[u][0], bound_points[u][1])
        end = (bound_points[u+1][0], bound_points[u+1][1])
        face_bound = cv2.line(face_bound, start, end, 255, 10)
        u += 1
start = (bound_points[len(bound_points)-1][0], bound_points[len(bound_points)-1][1])
end = (bound_points[0][0], bound_points[0][1])
face_bound = cv2.line(face_bound, start, end, 255, 10)
rs_new = np.ones_like(rs)*255
for y in range(len(target)):
    for x in range(len(target[0])):
        if face_bound[y][x] == 255:
            lower_x = x-5
            while lower_x < 0:
                lower_x += 1
            upper_x = x+5
            while upper_x >= len(target[0]):
                upper_x -= 1
            lower_y = y-5
            while lower_y < 0:
                lower_y += 1
            upper_y = y+5
            while upper_y >= len(target):
                upper_y -= 1
            splice = rs[lower_y:upper_y+1, lower_x:upper_x+1]
            if len(splice) == 0:
                continue
            median = np.median(splice)
            rs_new[y][x] = median
        else:
            rs_new[y][x] = rs[y][x]


# cv2.imshow("remapped", rs_new)
# cv2.imshow("face_blound", face_bound)
# cv2.waitKey(0)
# cv2.destroyAllWindows()       





final_l = rs_new + final_skin_details
final_l = cv2.medianBlur(final_l, 5)





merged = cv2.merge((final_l, final_a, final_b))
merged = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
cv2.imwrite("remapped.jpg", merged)


# LIPS MAKE UP




def histogram_equilize(image):
    equilised_img = np.zeros((image.shape[0], image.shape[1]))
    hist, _ = np.histogram(image.flatten(),256,[0,255])
    eq = hist.cumsum()/float(image.shape[0]*image.shape[1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equilised_img[i,j] = eq[image[i,j]] * 255
    return equilised_img.astype('uint8')


def gaussian(x, s, m):
    return 1/(math.sqrt(2*math.pi)*s) * math.e**(-0.5*(float(x-m)/s)**2)


def lip_makeup(dst_pts,src_pts,dst,src, sm):
    M=dst.copy()
    dst_l, dst_a, dst_b = cv2.split(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
    src_l, src_a, src_b = cv2.split(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
    dst_l = histogram_equilize(dst_l)
    src_l = histogram_equilize(sm)
    for p in dst_pts:
        qx = p[0]
        qy = p[1]
        maxfun = 0
        for q in src_pts:
            fun = gaussian( math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2), 1, 0) * gaussian( abs(dst_l[p[0],p[1]] - src_l[q[0],q[1]]), 1, 0)
            if(fun > maxfun):
                maxfun = fun
                qx = q[0]
                qy = q[1]
#         smc = cv2.merge
        M[p[0]][p[1]] = src[qx][qy]
    
    dst_l, dst_a, dst_b = cv2.split(cv2.cvtColor(dst, cv2.COLOR_BGR2LAB))
    M_l, M_a, M_b = cv2.split(cv2.cvtColor(M, cv2.COLOR_BGR2LAB))
    M = cv2.merge((dst_l, M_a, M_b))
    M = cv2.cvtColor(M, cv2.COLOR_LAB2BGR)
    return M





lip_points_src = []
lip_points_dst = []
for i in range(len(cmat)):
    for j in range(len(cmat[0])):
        if cmat[i][j] == 2:
            lip_points_src.append([i,j])
            lip_points_dst.append([i,j])
print(len(lip_points_src))        
Final = lip_makeup(lip_points_dst, lip_points_src, merged, warped_src, warp_smooth)
cv2.imwrite("lip.jpg", Final)


# XDOG TRANSFER




def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
    img1 = cv2.GaussianBlur(img,size,sigma)
    img2 = cv2.GaussianBlur(img,size,sigma*k)
    return (img1-gamma*img2)


def xdog(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
    aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
    for i in range(0,aux.shape[0]):
        for j in range(0,aux.shape[1]):
            if(aux[i,j] >= epsilon):
                aux[i,j] = 1
            else:
                ht = np.tanh(phi*(aux[i][j] - epsilon))
                aux[i][j] = 1 + ht
    return aux*255


xdog = xdog(target_l,sigma=0.9,k=200, gamma=0.98,epsilon=0.1,phi=5).astype('uint8')
# cv2.imshow("xdog", xdog)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

xdog_a = np.ones_like(xdog).astype('uint8')
xdog_b = np.ones_like(xdog).astype('uint8')
xdog_a, xdog_b = xdog_a*128, xdog_b*128
f_l, f_a, f_b = cv2.split(cv2.cvtColor(Final, cv2.COLOR_BGR2LAB))
x_a, x_b =  f_a - target_a, f_b - target_b
for i in range(len(xdog)):
    for j in range(len(xdog[0])):
        if mask[i][j] == 255:
            if int(xdog[i][j]) - 50 >= 0:
                xdog[i][j] -= 50
            xdog_a[i][j] = f_a[i][j]
            xdog_b[i][j] = f_b[i][j]

xdog = cv2.merge((xdog, xdog_a, xdog_b))
xdog = cv2.cvtColor(xdog, cv2.COLOR_LAB2BGR)
cv2.imwrite("xdog.jpg", xdog)

