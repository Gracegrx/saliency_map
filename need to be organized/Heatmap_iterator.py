# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:55:06 2017
This is for distract Region of Interest
@author: iris
"""
from importer_new import loadmat
from Heatmap.heatmap import Heatmap
#from heatmap import Heatmap
import numpy as np
from PIL import Image, ImageChops

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
data = loadmat('data/Basic_Mouth_new')
#data = loadmat('data/Basic_v0524')


class Heatmap_Iterator(object):
    def __init__(self, Slide_type = 'all', Week = 'all', Slide_num = 'all', ImId = 'all', Subject = 'all', m = 10, n = 10, ori_img = 'resources/10_week8.bmp', size_x = 302, size_y = 400, uniqueID = 'all'):
        self.Slide_type = Slide_type
        self.Week = Week
        self.Slide_num = Slide_num
        self.ImId = ImId
        self.Subject = Subject
        self.m = m
        self.n = n
        self.ori_img = ori_img
        self.size_x = size_x
        self.size_y = size_y
        self.uniqueID = uniqueID
        
        
    def load_subject(self, st_item, w_item, sn_item, img_item):
        subject = data['Data'][st_item][w_item][sn_item][img_item].keys()
        for subject_item in subject:
            if subject_item in self.Subject or self.Subject == 'all':
                temp = Heatmap(st_item, w_item, sn_item, img_item, subject_item, self.m, self.n, self.ori_img, self.size_x, self.size_y)
                if len(temp.get_uniqueID()) == 4 or len(temp.get_uniqueID()) == 5:
                    if temp.get_uniqueID() in self.uniqueID or self.uniqueID == 'all':
                        yield temp
            
        
    def load_img(self, st_item, w_item, sn_item):
        imgid = data['Data'][st_item][w_item][sn_item].keys()
        for img_item in imgid:
            if img_item in self.ImId or self.ImId == 'all':            
                for temp_img in self.load_subject(st_item, w_item, sn_item, img_item):
                    yield temp_img

    def load_slidenum(self, st_item, w_item):
        slidenum = data['Data'][st_item][w_item].keys()
        for sn_item in slidenum:
            if sn_item in self.Slide_num or self.Slide_num == 'all': 
                for temp_sn in self.load_img(st_item, w_item, sn_item):
                    yield temp_sn


    def load_week(self, st_item):
        week = data['Data'][st_item].keys()
        for w_item in week:
            if w_item in self.Week or self.Week == 'all':
                for temp_w in self.load_slidenum(st_item, w_item):
                    yield temp_w

        
    def load(self):
        slidetype = data['Data'].keys()
        st_list = []
        for st_item in slidetype:
            if st_item in self.Slide_type or self.Slide_type == 'all':
                st_list = [a for a in self.load_week(st_item)]
        #print len(st_list)
        return st_list
   
    def fixation_time(self):
        hs = self.load()
        time = 0.0
        for h in hs:
            if h.get_fixationtime() != -1:
                time += h.get_fixationtime_mouth()
        return time
       
    def avg_fixationtime_mouth(self):
        hs = self.load()
        time = 0.0
        count = 0.0
        for h in hs:
            if h.get_fixationtime_mouth() != -1:
                time += h.get_fixationtime_mouth()
                count += 1
        if count == 0:
            return 0
        avg_time = time/count
        return avg_time
        
    def avg_rfixationtime_mouth(self):
        hs = self.load()
        rtime = 0.0
        count = 0.0
        for h in hs:
            if h.get_rfixationtime_mouth() != -1:
                rtime += h.get_rfixationtime_mouth()
                count += 1
        if count == 0:
            return 0
        avg_rtime = rtime/count
        return avg_rtime
       

    def get_ori_img(self):
        hs = self.load()
        print hs[0].get_region()
        print hs[0].slidenum
        print hs[0].ImgID
        img1 = hs[0].get_ori_img()
        return img1

        
    def get_pro_matrix(self, matrix):
        hs = self.load()
        img = hs[0].adjust_size(matrix)
        return img

    def get_ori_size_matrix(self):
        hs = self.load()
        #print len(hs)
        res = np.zeros(shape=(self.n, self.m))
        for h in hs:
            #print len(h.get_ori_heat_matrix()),len(h.get_ori_heat_matrix()[0])
            #print h.get_ori_heat_matrix()
            res += h.get_ori_heat_matrix()
        if len(hs) != 0:
            img2_pro = res/len(hs)
        else:
            img2_pro = res

        file=open('pro_matrix.txt','w')
        for l in img2_pro:
            file.write(str(l))
        file.close()

        return img2_pro

    def get_diff(self, other):
        img1 = self.get_ori_size_matrix()
        #print 'img1 = ', img1
        img2 = other.get_ori_size_matrix()
        #print 'img2 = ', img2
        diff = img1-img2
        return diff

        
    def show_result(self):
        img1 = self.get_ori_img()
        img2_pro = self.get_pro_matrix(self.get_ori_size_matrix())
        rangee = np.amax(img2_pro)
        temp_matrix = (255/rangee)*np.ones(shape = (self.size_y, self.size_x))
        img2 = temp_matrix*img2_pro
        img2 = Image.fromarray(np.uint8(img2))     
        img2 = np.matrix(img2.getdata()).reshape(self.size_y, self.size_x)
#        final_matrix = img1*0.7+img2*0.3
        final_matrix = img1*0.4+img2*0.6
        img = Image.fromarray(np.uint8(final_matrix))
        img.show()

    def show_result_color(self):
        img1 = self.get_ori_img()
        img2_pro = self.get_pro_matrix(self.get_ori_size_matrix())
        rangee = np.amax(img2_pro)
        temp_matrix = (255 / rangee) * np.ones(shape=(self.size_y, self.size_x))
        img2 = temp_matrix * img2_pro
        img2 = Image.fromarray(np.uint8(img2))
        img2 = np.matrix(img2.getdata()).reshape(self.size_y, self.size_x)
        #        final_matrix = img1*0.7+img2*0.3
        final_matrix = img1 * 0.4 + img2 * 0.6
        img = Image.fromarray(np.uint8(final_matrix))
        img = img.convert("RGB")
        img.save('temp.jpg')
        img_new = mpimg.imread('temp.jpg')
        lum_img = img_new[:,:,0]
        #lum_img = img_new
        plt.imshow(lum_img)
        imgplot = plt.imshow(lum_img)
        plt.colorbar()
        plt.show()
        #img.show()
        
        
    def show_compare(self, other):
        diff = self.get_diff(other)
        adjust_diff_ori = self.get_pro_matrix(diff)
        file=open('diff.txt','w')
        for l in adjust_diff_ori:
            file.write(str(l))
        file.close()        
        rangee = np.amax(adjust_diff_ori)-np.amin(adjust_diff_ori)
        temp_matrix = (255/rangee)*np.ones(shape = (self.size_y, self.size_x))
        matrix = np.amin(adjust_diff_ori)*np.ones(shape = (self.size_y, self.size_x))
        adjust_diff = adjust_diff_ori-matrix
        adjust_diff = adjust_diff*temp_matrix
        img = Image.fromarray(255*np.uint8(adjust_diff))
#        img.show()
        img = img.convert("RGB")
        #maximum = np.amax(adjust_diff_ori)
        #minimum = np.amin(adjust_diff_ori)
        maximum = 0.05
        minimum = -0.05
        for i in range(len(adjust_diff)):
            for j in range(len(adjust_diff[i])):
                if adjust_diff_ori[i][j] >= 0:
                    num =min(255, int(255*(1-(maximum-adjust_diff_ori[i][j])/maximum)))
                    img.putpixel((j,i), (num,0,0))
                else:
                    num =min(255, int(255*(1-(minimum-adjust_diff_ori[i][j])/minimum)))
                    img.putpixel((j,i), (0,0,num))
#        img.show()
        
        hs = self.load()
        region = hs[0].get_region()
        im = Image.open(self.ori_img)
        img1 = im.crop(region)                
        out = ImageChops.multiply(img1, img)
        out.show()




    '''
    def decide(self, matrix):
        sample = self.get_ori_size_matrix()        
        value = 0
        for i in range(len(sample)):
            for j in range(len(sample[i])):
                if matrix[i][j] >= 2e-03:
                    value += sample[i][j]
                elif matrix[i][j] < -2e-03:
                    value -= sample[i][j]
        return value
    '''

    def get_region(self, matrix):
        sample = self.get_ori_size_matrix()
        value = 0
        for i in range(len(sample)):
            for j in range(len(sample[i])):
                value += sample[i][j]* matrix[i][j]
        return value

    def show_roi(self, matrix):
        large = self.get_pro_matrix(matrix)
        img = Image.fromarray(255*np.uint8(large))
        img = img.convert("RGB")
        for i in range(len(large)):
            for j in range(len(large[i])):
                if large[i][j] >= 2e-03:
                    img.putpixel((j,i), (128,0,0))
                elif large[i][j] < -2e-03:
                    img.putpixel((j,i), (0,0,128))
        img.show()

    def show_roi_grey(self, matrix, boundary):
        img1 = self.get_ori_img()
        large = self.get_pro_matrix(matrix)
        img = Image.fromarray(255*np.uint8(large))
        for i in range(len(large)):
            for j in range(len(large[i])):
                if large[i][j] >= boundary:
                    img.putpixel((j,i), 128)
        img2 = np.array(img)
        final_matrix = img1 * 0.4 + img2 * 0.6
        img = Image.fromarray(np.uint8(final_matrix))
        img.show()


    def get_mean_GD1(self):
        hs = self.load()
        count = 0
        value = 0
        for item in hs:
            if len(item.get_GD()) != 0:
                value += item.get_GD()[0]
                count += 1.0
        if count > 0:
            return value/count
        else:
            return 0

    def get_mean_GD2(self):
        hs = self.load()
        count = 0
        value = 0
        for item in hs:
            if len(item.get_GD()) > 1:
                value += item.get_GD()[1]
                count += 1.0
        if count > 0:
            return value/count
        else:
            return 0

    def get_FFBI(self):
        hs = self.load()
        count = 0
        value = 0
        for item in hs:
            if item.get_FFBI() >= 0:
                value += item.get_FFBI()
                count += 1.0
        if count > 0:
            return value/count
        else:
            return 0

    def get_FFWI(self):
        hs = self.load()
        count = 0
        value = 0
        for item in hs:
            if item.get_FFWI() >= 0:
                value += item.get_FFWI()
                count += 1.0
        if count > 0:
            return value/count
        else:
            return 0


    def show_compare_matrix(self, input):
        adjust_diff_ori = self.get_pro_matrix(input)
        rangee = np.amax(adjust_diff_ori) - np.amin(adjust_diff_ori)
        temp_matrix = (255 / rangee) * np.ones(shape=(self.size_y, self.size_x))
        matrix = np.amin(adjust_diff_ori) * np.ones(shape=(self.size_y, self.size_x))
        adjust_diff = adjust_diff_ori - matrix
        adjust_diff = adjust_diff * temp_matrix
        img = Image.fromarray(255 * np.uint8(adjust_diff))
        img = img.convert("RGB")
        maximum = np.amax(adjust_diff_ori)
        minimum = np.amin(adjust_diff_ori)
        for i in range(len(adjust_diff)):
             for j in range(len(adjust_diff[i])):
                  if adjust_diff_ori[i][j] >= 0:
                        num = min(255, int(255 * (1 - (maximum - adjust_diff_ori[i][j]) / maximum)))
                        img.putpixel((j, i), (num, 0, 0))
                  else:
                        num = min(255, int(255 * (1 - (minimum - adjust_diff_ori[i][j]) / minimum)))
                        img.putpixel((j, i), (0, 0, num))

        hs = self.load()
        region = hs[0].get_region()
        im = Image.open(self.ori_img)
        img1 = im.crop(region)
        out = ImageChops.multiply(img1, img)
        out.show()





#example = Heatmap_Iterator(Slide_type = 'TEST_IAPS', Week = 'EDWeek2_TGH', Subject = group_remitters, ori_img = '13.bmp', size_x = 590, size_y = 442, uniqueID = '4700')
#example.show_result()


    
   
#h = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'])
#print h.load_slidetype()
#print Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Slide_num = ['Slide_Num10'])
