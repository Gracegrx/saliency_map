# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 13:23:17 2017

@author: iris
"""
#into a class
import numpy as np
from importer_new import loadmat
from PIL import Image

#data = loadmat('Basic_Mouth_new')
data = loadmat('data/Basic_Mouth_new')
#data = loadmat('data/Basic_v0524')
class Heatmap(object):
    #use to generate the heatmap for the certain image
    def __init__(self, Slidetype, week, slidenum, ImgID, Subject, m, n, ori_img, size_x, size_y):
        self.Slidetype = Slidetype
        self.week = week
        self.slidenum = slidenum
        self.ImgID = ImgID
        self.Subject = Subject
        self.m = m
        self.n = n
        self.ori_img = ori_img
        self.size_x = size_x
        self.size_y = size_y
    
    def get_pos(self):
        temp_x = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['FixationPositionsX']
        temp_y = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['FixationPositionsY']
        if type(temp_x) != np.ndarray:
            temp_x = []
        if type(temp_x) != np.ndarray:
            temp_y = []
        return temp_x, temp_y
        
    def get_region(self):
        loc_x = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['Loc_x']
        loc_y = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['Loc_y']
        region = [loc_x, loc_y, loc_x+self.size_x, loc_y+self.size_y]
        return region
        
        
    def get_uniqueID(self):
        ID = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['uniqueImgIDs']
        return ID

    def get_fixationtime_mouth(self):
        fixationtime_mouth = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['FixationTime_ms_Mouth']
        return fixationtime_mouth
        
    def get_fixationtime(self):
        fixationtime = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['FixationTime_ms']
        return fixationtime  
                
    def get_rfixationtime_mouth(self):
        if self.get_fixationtime_mouth() != -1 and self.get_fixationtime != -1:
            return self.get_fixationtime_mouth()/self.get_fixationtime
        else:
            return -1
    
    def get_FFBI(self):
        FFBI = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['FixFrequencyBetweenImages']
        if FFBI >= 0:
            return FFBI
        else:
            return -1
            
    def get_GD(self):
        GD = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['GlanceDuration_ms']
        if GD == -1:
            return[]
        else:
            return GD

    def get_FFWI(self):
        FFWI = data['Data'][self.Slidetype][self.week][self.slidenum][self.ImgID][self.Subject]['FixFreqWithinImage']
        if FFWI >= 0:
            return FFWI
        else:
            return -1

#division here is the main heatmap matrix.        
    def count_fixations(self):
        #divide into m parts in the row and n parts in the col
        division = np.zeros(shape = (self.n, self.m))
        x, y = self.get_pos()
        region = self.get_region()
        if len(x) == 0:
            return division
        for i in range(len(x)):
            if region[1] <y[i]<region[3] and region[0]<x[i]<region[2]:
                division[(y[i]-region[1])*self.n/(region[3]-region[1])][(x[i]-region[0])*self.m/(region[2]-region[0])] += 1.0/(len(x)-2)
#        print np.amax(division)
#        print division
        return division
       
    def adjust_size(self, matrix):
        #make a matrix that represent the density of fixiation with the same size of a img.
        res = []
        region = self.get_region()
        margin = np.zeros(region[2]-region[0])
        fil = (region[3]-region[1]) % self.n
        d = 0
        while d < fil/2:
            res.append(margin)
            d += 1
        for i in range(len(matrix)):
            temp = []
            em = (region[2]-region[0]) % self.m
            c = 0
            while c < em/2:
                temp.append(0)
                c += 1
            g = 0
            for j in matrix[i]:
                k = 0
                while k < (region[2]-region[0])/self.m:
                    temp.append(j)
                    k += 1
            c = 0
            while c < (em+1)/2:
                temp.append(0)
                c += 1
            while g < (region[3]-region[1])/self.n:
                res.append(temp)
                g += 1
        d = 0
        while d < (fil+1)/2:
            res.append(margin)
            d += 1
                
        return res
    
    def get_ori_img(self):
        region = self.get_region()
        im = Image.open(self.ori_img)
        img1 = im.crop(region).convert("L")
        img1_data = np.matrix(img1.getdata()).reshape(region[3]-region[1], region[2]-region[0])
        return img1_data

    def get_heat_matrix(self):
        img2 = self.adjust_size(self.count_fixations())
        return img2
                
    def get_ori_heat_matrix(self):
        ori_size_matrix = self.count_fixations()
        return ori_size_matrix
        
#final matrix here is matrix for show.        
    def get_final_matrix(self):
        region = self.get_region()
        img1 = self.get_ori_img()
        img2 = self.get_heat_matrix()
        temp_matrix = 255*np.ones(shape = (self.size_y, self.size_x))
        img2 = temp_matrix*img2
        img2 = Image.fromarray(255*np.uint8(img2))     
        img2 = np.matrix(img2.getdata()).reshape(region[3]-region[1], region[2]-region[0])
        final_matrix = img1*0.4+img2*0.6
        temp_matrix = 255/(np.amax(final_matrix)-np.amin(final_matrix))*np.ones(shape = (self.size_y, self.size_x))
        final_matrix = temp_matrix*final_matrix
        print np.amax(final_matrix)
        return final_matrix
        
    def show_img(self, matrix):
        img = Image.fromarray(np.uint8(matrix))
        img.show()
        
    def show_heat_map(self):
        self.show_img(self.get_final_matrix())
        
        
    







#a = Heatmap('TEST_KDEF', 'EDWeek2_TGH', 'Slide_Num10', 'ImId1', 'Subject10D', 10, 10, 'resources/22_week3.bmp', 302, 400)
#print a.get_uniqueID()    