from Heatmap.Heatmap_iterator import Heatmap_Iterator
import numpy as np
from PIL import Image
from pylab import *

group_ctrl = ['Subject27C', 'Subject28C', 'Subject29C', 'Subject30C', 'Subject31C', 'Subject32C', 'Subject33C', 'Subject34C', 'Subject35C', 'Subject36C', \
                    'Subject37C', 'Subject38C', 'Subject39C', 'Subject42C','Subject46c', 'Subject50C', 'Subject51C', 'Subject56C', 'Subject58C', 'Subject59C', 'Subject60C', \
                    'Subject61C', 'Subject62C', 'Subject63C', 'Subject66C', 'Subject69C', ]


#find index
def find_center(matrix):
    cur = np.amax(matrix)
    for i in range(len(matrix)):
        if cur in matrix[i]:
            num = list(matrix[i]).index(cur)
            return [i, num]


def find_region(cur, matrix, region, num):
    seq = [cur]
    summ = 0
    while summ < target_prob:
        summ += matrix[cur[0], cur[1]]
        matrix[cur[0]][cur[1]] = 0
        region[cur[0]][cur[1]] = num
        temp = [[cur[0]-1, cur[1]-1], [cur[0]-1, cur[1]], [cur[0]-1, cur[1]+1], \
                [cur[0], cur[1]-1], [cur[0], cur[1]+1], \
                [cur[0]+1, cur[1]-1], [cur[0]+1, cur[1]], [cur[0]+1, cur[1]+1]]
        temp = [[i,j] for [i,j] in temp if i >=0 and i < len(matrix) and j >= 0 and j < len(matrix[0])]
        candidates = {}
        for [i,j] in temp:
            candidates[i,j] = matrix[i][j]
        curr = np.max(candidates.values())
        if curr == 0:
            print "region not large enough"
            return seq
        cur = [i for i in candidates.keys() if candidates[i] == curr][0]
        seq.append(cur)
    matrix[cur[0]][cur[1]] = 0
    region[cur[0]][cur[1]] = num
    return seq

def find_square_region(cur, matrix, region, num):
    seq = [cur]
    temp_boundary = [cur[0]-1, cur[0]+1, cur[1]-1, cur[1]+1]
    summ = 0
    while summ < target_prob:
        cc = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i >= temp_boundary[0] and i <= temp_boundary[1] and j >= temp_boundary[2] and j <= temp_boundary[3] and region[i][j] == 0:
                    summ += matrix[i, j]
                    matrix[i][j] = 0
                    region[i][j] = num
                    seq.append([i,j])
                    cc += 1
        temp_boundary[0] = max(temp_boundary[0]-1, 0)
        temp_boundary[1] = min(temp_boundary[1]+1, len(matrix))
        temp_boundary[2] = max(temp_boundary[2]-1, 0)
        temp_boundary[3] = min(temp_boundary[3]+1, len(matrix[0]))
        if cc == 0:
            return seq
    return seq



def start(matrix, region):
    count = 1
    result = []
    while count < num:
        cur = find_center(matrix)
        seq = find_square_region(cur, matrix, region, count)
        count += 1
        result.append(seq)
    return result, region

def adjust_size(matrix, size, m, n):
    # make a matrix that represent the density of fixiation with the same size of a img.
    res = []
    margin = np.zeros(size[0])
    fil = size[1] % n
    d = 0
    while d < fil / 2:
        res.append(margin)
        d += 1
    for i in range(len(matrix)):
        temp = []
        em = size[0] % m
        c = 0
        while c < em / 2:
            temp.append(0)
            c += 1
        g = 0
        for j in matrix[i]:
            k = 0
            while k < size[0] / m:
                temp.append(j)
                k += 1
        c = 0
        while c < (em + 1) / 2:
            temp.append(0)
            c += 1
        while g < size[1] / n:
            res.append(temp)
            g += 1
    d = 0
    while d < (fil + 1) / 2:
        res.append(margin)
        d += 1

    return res


#def show_regions(res):
#    color_dic = ['r', 'b', 'y', 'g', 'c']
#    for region in res:
'''
num = 10
target_prob = 1.0/num
alist = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_ctrl, m=100, n=100)
img1 = alist.get_ori_img()
img2 = img1.tolist()
size = [len(img2[0]), len(img2)]
matrix = alist.get_ori_size_matrix()
region = np.zeros(shape = (len(matrix), len(matrix[0])))
res, region = start(matrix, region)
rangee = np.amax(region)
temp_matrix = (255 / rangee) * np.ones(shape=(len(matrix), len(matrix[0])))
region = region * temp_matrix
region = adjust_size(region, size, 100, 100)
region = np.array(region)
img = Image.fromarray(region)
img.show()
img2 = np.matrix(img.getdata()).reshape(size[1], size[0])
#final_matrix = img1*0.7+img2*0.3
final_matrix = img1*0.4+img2*0.6
img = Image.fromarray(np.uint8(final_matrix))
img.show()
#imshow(region)
'''


def mouth_model(m, n, matrix, region, prob, order):
    box = [0.25, 0.75, 0.75]
    start = box[0]*m
    end = box[1]*m
    y = [int(box[2] * n), int(box[2] * n)]
    summ = 0
    while summ < prob:
        for j in range(int(start), int(end)+1):
            for number in y:
                summ += matrix[number][j]
                region[number][j] = order
                matrix[number][j] = 0
        y = [y[0]-1, y[1]+1]
    print 'mouth'
    return y[0]


def inner_eyes_model(m, n, matrix, region, prob, order):
    box = [0.25, 0.75, 0.35]
    start = box[0] * m
    end = box[1] * m
    y = [int(box[2] * n), int(box[2] * n)]
    summ = 0
    while summ < prob:
        for j in range(int(start), int(end)+1):
            for number in y:
                summ += matrix[number][j]
                region[number][j] = order
                matrix[number][j] = 0
        y = [y[0] - 1, y[1] + 1]
    print 'eye'
    return y[1]


def nose_model(box, m, n, matrix, region, prob, order):
    start = box[0]
    end = box[1]
    x = [int(box[2] * m), int(box[2] * m)]
    summ = 0
    while summ < prob:
        for i in range(int(start), int(end)+1):
            for number in x:
                summ += matrix[i][number]
                region[i][number] = order
                matrix[i][number] = 0
        x = [x[0] - 1, x[1] + 1]
    print 'nose'
    return region

def left(matrix, region, order):
    for i in range(len(matrix)/2):
        for j in range(len(matrix[0])):
            if region[j][i] == 0:
                region[j][i] = order
def right(matrix, region, order):
    for i in range(len(matrix)/2, len(matrix)):
        for j in range(len(matrix[0])):
            if region[j][i] == 0:
                region[j][i] = order

prob = 0.2
alist = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_ctrl, m=100, n=100)
img1 = alist.get_ori_img()
img2 = img1.tolist()
size = [len(img2[0]), len(img2)]
matrix = alist.get_ori_size_matrix()
region = np.zeros(shape = (len(matrix), len(matrix[0])))
lower = mouth_model(100, 100, matrix, region, prob, 1)
upper = inner_eyes_model(100, 100, matrix, region, prob, 2)
box = [upper, lower, 0.5]
region = nose_model(box, 100, 100, matrix, region, prob, 3)
left(matrix, region, 4)
right(matrix, region, 5)
#print region
rangee = np.amax(region)
temp_matrix = (255 / rangee) * np.ones(shape=(len(matrix), len(matrix[0])))
region = region * temp_matrix
region = adjust_size(region, size, 100, 100)
region = np.array(region)
img = Image.fromarray(region)
img.show()
img2 = np.matrix(img.getdata()).reshape(size[1], size[0])
#final_matrix = img1*0.7+img2*0.3
final_matrix = img1*0.4+img2*0.6
img = Image.fromarray(np.uint8(final_matrix))
img.show()