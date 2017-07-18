# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:51:00 2017

@author: iris
"""
import matplotlib
matplotlib.use('Qt4Agg')

import math
from example_heatmap import *
from Heatmap.heatmap import Heatmap
from Heatmap.Heatmap_iterator import Heatmap_Iterator
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pylab as pl
import random
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import ShuffleSplit
    
    
group_re = ['Subject1D', 'Subject3D', 'Subject7D', 'Subject8D', 'Subject10D', 'Subject12D', 'Subject13D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject22D', 'Subject25D', \
                'Subject41D', 'Subject47D', 'Subject54D', 'Subject68D', 'Subject73D', 'Subject75D', 'Subject83D', 'Subject85D', 'Subject86D', 'Subject90D', 'Subject93D']

#abandon 'Subject2D'
group_non = ['Subject4D','Subject5D', 'Subject6D', 'Subject9D', 'Subject11D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject23D', \
                'Subject24D', 'Subject64D', 'Subject70D', 'Subject71D', 'Subject72D', 'Subject80D', 'Subject81D', 'Subject84D', 'Subject88D', 'Subject96D']
                
group_remitters = ['Subject1D', 'Subject3D', 'Subject7D', 'Subject10D', 'Subject12D', 'Subject13D', 'Subject22D', \
                'Subject41D', 'Subject47D', 'Subject54D', 'Subject68D', 'Subject73D', 'Subject75D', 'Subject83D', 'Subject85D', 'Subject86D', 'Subject93D']
                
group_renon = ['Subject8D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject25D', 'Subject90D']

group_nonremitters = ['Subject2D', 'Subject4D','Subject5D', 'Subject6D', 'Subject9D', 'Subject11D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject23D', \
                'Subject24D', 'Subject64D', 'Subject70D', 'Subject71D', 'Subject72D', 'Subject80D', 'Subject81D', 'Subject84D', 'Subject88D', 'Subject96D', 'Subject8D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject25D', 'Subject90D']

group_depress = ['Subject1D', 'Subject3D', 'Subject7D', 'Subject8D', 'Subject10D', 'Subject12D', 'Subject13D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject22D', 'Subject25D', \
                'Subject41D', 'Subject47D', 'Subject54D', 'Subject68D', 'Subject73D', 'Subject75D', 'Subject83D', 'Subject85D', 'Subject86D', 'Subject90D', 'Subject93D', 'Subject2D', \
                'Subject4D','Subject5D', 'Subject6D', 'Subject9D', 'Subject11D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject23D', \
                'Subject24D', 'Subject64D', 'Subject70D', 'Subject71D', 'Subject72D', 'Subject80D', 'Subject81D', 'Subject84D', 'Subject88D', 'Subject89D', 'Subject96D', 'Subject97D']
                
group_uni = ['Subject1D', 'Subject10D', 'Subject11D', 'Subject12D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject19D', 'Subject2D', 'Subject20D', \
                 'Subject21D', 'Subject22D', 'Subject23D', 'Subject24D', 'Subject27C', 'Subject28C', 'Subject29C', 'Subject3D', 'Subject30C', 'Subject31C', 'Subject32C', 'Subject33C', \
                 'Subject34C', 'Subject35C', 'Subject36C', 'Subject37C', 'Subject38C', 'Subject39C', 'Subject4D', 'Subject40R', 'Subject41D', 'Subject42C', 'Subject43R', \
                 'Subject45R', 'Subject46c', 'Subject47D', 'Subject48R', 'Subject49R', 'Subject5D', 'Subject50C', 'Subject51C', 'Subject52R', 'Subject53R', \
                 'Subject54D', 'Subject55R', 'Subject56C', 'Subject57R', 'Subject58C', 'Subject59C', 'Subject6D', 'Subject60C', 'Subject61C', 'Subject62C', 'Subject63C', \
                 'Subject64D', 'Subject65R', 'Subject66C', 'Subject67R', 'Subject68D', 'Subject69C', 'Subject7D', 'Subject70D', 'Subject71D', 'Subject72D', \
                 'Subject73D', 'Subject74R', 'Subject75D', 'Subject76R', 'Subject77R', 'Subject79R', 'Subject8D', 'Subject80D', 'Subject81D', 'Subject83D', 'Subject84D', \
                 'Subject85D', 'Subject86D', 'Subject87R', 'Subject88D', 'Subject89D', 'Subject9D', 'Subject90D', 'Subject91R', 'Subject92R', 'Subject93D', 'Subject94R', 'Subject95R', 'Subject97D']
                 
group_undepress = ['Subject27C', 'Subject28C', 'Subject29C', 'Subject30C', 'Subject31C', 'Subject32C', 'Subject33C', \
                 'Subject34C', 'Subject35C', 'Subject36C', 'Subject37C', 'Subject38C', 'Subject39C', 'Subject40R', 'Subject42C', 'Subject43R', \
                 'Subject45R', 'Subject46c', 'Subject48R', 'Subject49R', 'Subject50C', 'Subject51C', 'Subject52R', 'Subject53R', \
                 'Subject55R', 'Subject56C', 'Subject57R', 'Subject58C', 'Subject59C', 'Subject60C', 'Subject61C', 'Subject62C', 'Subject63C', \
                 'Subject65R', 'Subject66C', 'Subject67R', 'Subject69C', 'Subject74R', 'Subject76R', 'Subject77R', 'Subject79R', \
                 'Subject87R', 'Subject91R', 'Subject92R', 'Subject94R', 'Subject95R']
                 
group_remitted = ['Subject40R', 'Subject43R', 'Subject45R', 'Subject48R', 'Subject49R', 'Subject52R', 'Subject53R', \
                 'Subject55R', 'Subject57R', 'Subject65R', 'Subject67R', 'Subject74R', 'Subject76R', 'Subject77R', 'Subject79R', \
                 'Subject87R', 'Subject91R', 'Subject92R', 'Subject94R', 'Subject95R']
    
group_bi = ['BD004D', 'BD005R', 'BD006R', 'BD010D', 'BD011D', 'BD013R', 'BD014D', 'BD015D', 'BD016R', 'BD017D', 'BD018D', 'BD019R', 'BD020R', 'BD021D', 'BD022D', 'BD024D', 'BD025D', \
                'BD026D', 'BD027D', 'BD028D', 'BD029D', 'BD030D', 'BD031D', 'BD033D', 'BD034R', 'BD035D', 'BD036D', 'BD037R', 'BD038D', 'BD040R', 'BD042R', 'BD043R', 'BD044R', 'BD046R', 'BD047R', \
                'BD001D', 'BD002R', 'BD003R', 'BD007D', 'BD008D', 'BD009D', 'BD012R', 'BD023R', 'BD032D', 'BD039R', 'BD041R']
                
                
group_ctrl = ['Subject27C', 'Subject28C', 'Subject29C', 'Subject30C', 'Subject31C', 'Subject32C', 'Subject33C', 'Subject34C', 'Subject35C', 'Subject36C', \
                    'Subject37C', 'Subject38C', 'Subject39C', 'Subject42C','Subject46c', 'Subject50C', 'Subject51C', 'Subject56C', 'Subject58C', 'Subject59C', 'Subject60C', \
                    'Subject61C', 'Subject62C', 'Subject63C', 'Subject66C', 'Subject69C', ]           
                
                
happy_face = ['F02HA', 'F01HA', 'M05HA', 'M10HA', 'F33HA', 'F30HA', 'M23HA', 'M31HA', 'F03HA', 'F26HA', 'M35HA', 'M12HA', 'F16HA', 'F05HA', 'M06HA', 'M24HA', 'F31HA', 'F13HA', 'M13HA', \
                    'M11HA', 'F14HA', 'F09HA', 'M22HA', 'M25HA', 'F07HA', 'F11HA', 'F17HA']
sad_face = ['F02SA', 'F01SA', 'M05SA', 'M10SA', 'F33SA', 'F30SA', 'M23SA', 'M31SA', 'F03SA', 'F26SA', 'M35SA', 'M12SA', 'F16SA', 'F05SA', 'M06SA', 'M24SA', 'F31SA', 'F13SA', 'M13SA', \
                    'M11SA', 'F14SA', 'F09SA', 'M22SA', 'M25SA', 'F07SA', 'F11SA', 'F17SA']
                    
                    



def get_para_GD(slide_type, group, week):
    GD = []
    for subject in group:
        temp = Heatmap_Iterator(Slide_type = slide_type, Week = week, Subject = subject)
        GD.append(temp.get_mean_GD1() + temp.get_mean_GD2())
    return GD

def get_para_FFBI(slide_type, group, week):
    FFBI = []
    for subject in group:
        temp = Heatmap_Iterator(Slide_type = slide_type, Week = week, Subject = subject)
        FFBI.append(temp.get_FFBI())
    return FFBI

def get_FFBI_diff(slide_type, group, week1, week2):
    res = []
    for subject in group:
        temp1 = Heatmap_Iterator(Slide_type = slide_type, Week = week1, Subject = subject)
        temp2 = Heatmap_Iterator(Slide_type = slide_type, Week = week2, Subject = subject)
        if temp1.get_FFBI()!= 0 and temp2.get_FFBI()!= 0:
            res.append(temp1.get_FFBI()-temp2.get_FFBI())
    return res

def get_para_FFWI(slide_type, group, week):
    FFWI = []
    for subject in group:
        temp = Heatmap_Iterator(Slide_type = slide_type, Week = week, Subject = subject)
        FFWI.append(temp.get_FFWI())
    return FFWI


def get_FFWI_diff(slide_type, group, week1, week2):
    res = []
    for subject in group:
        temp1 = Heatmap_Iterator(Slide_type = slide_type, Week = week1, Subject = subject)
        temp2 = Heatmap_Iterator(Slide_type = slide_type, Week = week2, Subject = subject)
        if temp1.get_FFWI()!= 0 and temp2.get_FFWI()!= 0:
            res.append(temp1.get_FFWI()-temp2.get_FFWI())
    return res



def time_diff(group, week):
    value = []
    for subject in group:
        happy = 0.0
        sad = 0.0
        temp1 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = week, Subject = subject, uniqueID = happy_face)
        temp2 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = week, Subject = subject, uniqueID = sad_face)
        happy += temp1.fixation_time()
        sad += temp2.fixation_time()
        if happy + sad == 0:
            rhappy = 0
            rsad = 0
        else:
            rhappy = happy/(happy+sad)
            rsad = sad/(happy+sad) 
        value.append(temp1.mouth()*rhappy-temp2.mouth()*rsad)
    value = np.array(value)
    value = value[~np.isnan(value)]
#    print value
    return value

#calculate the difference of bands between week and week2, output is a value for the difference
def band_diff(n1, n2, week, group):
    value = []
    for subject in group:
        res = 0
        week2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=subject, m=1, n=10)
        week3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=week, Subject=subject, m=1, n=10)
        for num in n1:
            res += week3.get_diff(week2)[num]
        for num in n2:
            res -= week3.get_diff(week2)[num]
        value.append(res[0])
    value = np.array(value)
    #value = re_value[~np.isnan(value)]
    return value

def diff(group, week):
    value = []
    for subject in group:
        temp1 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = week, Subject = subject, uniqueID = happy_face, m = 10, n = 10)
        temp2 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = week, Subject = subject, uniqueID = sad_face, m = 10, n = 10)
        value.append(temp1.mouth()-temp2.mouth())
    value = np.array(value)
    value = value[~np.isnan(value)]
#    print value
    return value


def get_IAPS_value_DC():
    remitters = np.zeros(len(group_remitters))
    non_responders = np.zeros(len(group_non))
    count_re = np.zeros(len(group_remitters))
    count_non = np.zeros(len(group_non))
    for id in unique_ids:
        matrix = region_of_interest('13', id)
        for i in range(len(group_remitters)):
            temp = get_fixation_value(matrix, '13', id, group_remitters[i])
            if temp != 'nan':
                remitters[i] += temp
                if temp != 0:
                    count_re[i] += 1
        for i in range(len(group_non)):
            temp = get_fixation_value(matrix, '13', id, group_non[i])
            if temp != 'nan':
                non_responders[i] += temp
                if temp != 0:
                    count_non[i] += 1
    #print count_re
    #print count_non
    #for i in range(len(group_remitters)):
    #    remitters[i] = remitters[i]/count_re[i]
    #for i in range(len(group_non)):
    #    non_responders[i] = non_responders[i] / count_non[i]
    remitters = np.array(remitters)
    non_responders = np.array(non_responders)
    remitters = remitters[~np.isnan(remitters)]
    non_responders = non_responders[~np.isnan(non_responders)]
    return remitters, non_responders

def get_IAPS_value_C():
    file = open('matrix.txt', 'w')
    remitters = np.zeros(shape = (len(group_remitters), len(unique_ids)))
    non_responders = np.zeros(shape = (len(group_non), len(unique_ids)))
    count = 0
    for id in unique_ids:
        matrix = region_of_interest('13', id)
        # print matrix
        for i in range(len(group_remitters)):
            temp = get_fixation_value(matrix, '13', id, group_remitters[i])
            if temp != 'nan':
                #print temp
                remitters[i][count] += temp
            #print remitters[i][count]
        for i in range(len(group_non)):
            temp = get_fixation_value(matrix, '13', id, group_non[i])
            if temp != 'nan':
                non_responders[i][count] += temp
            #print non_responders[i][count]
        count += 1
    #remitters = np.array(remitters)
    #non_responders = np.array(non_responders)
    #remitters = remitters[~np.isnan(remitters)]
    #non_responders = non_responders[~np.isnan(non_responders)]
    file.write('remitters: ')
    for i in range(len(group_remitters)):
        print remitters[i]
        file.write(remitters[i])
    file.write('non-responders: ')
    for i in range(len(group_non)):
        print non_responders[i]
        file.write(non_responders[i])
    file.close()
    return remitters, non_responders

def get_IAPS_value_C_T():
    file = open('matrix_T.txt', 'w')
    remitters = np.zeros(shape = (len(unique_ids), len(group_remitters)))
    non_responders = np.zeros(shape = (len(unique_ids), len(group_non)))
    count = 0
    for id in unique_ids:
        matrix = region_of_interest('13', id)
        # print matrix
        for i in range(len(group_remitters)):
            temp = get_fixation_value(matrix, '13', id, group_remitters[i])
            if temp != 'nan':
                #print temp
                remitters[count][i] += temp
            #print remitters[i][count]
        for i in range(len(group_non)):
            temp = get_fixation_value(matrix, '13', id, group_non[i])
            if temp != 'nan':
                non_responders[count][i] += temp
            #print non_responders[i][count]
        count += 1
    #remitters = np.array(remitters)
    #non_responders = np.array(non_responders)
    #remitters = remitters[~np.isnan(remitters)]
    #non_responders = non_responders[~np.isnan(non_responders)]
    file.write('remitters: ')
    for i in range(len(unique_ids)):
        print remitters[i]
        file.write(remitters[i])
    file.write('non-responders: ')
    for i in range(len(unique_ids)):
        print non_responders[i]
        file.write(non_responders[i])
    file.close()
    return remitters, non_responders


def NN(para, label, nx, ny):
    X = para[:nx]
    print para
    y = label[:ny]
    print y
    clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,2), random_state = 1)
    clf.fit(X,y)       
       
    output = clf.predict(para)
    count = 0.0
    for i in range(len(output)):
        if output[i] == label[i]:
            count += 1
    accuracy = count/len(output)
    print 'train accuracy for NN = ', accuracy
    
    output_test = clf.predict(para[nx:])
    key = label[ny:]
    count = 0.0
    for i in range(len(output_test)):
        if output_test[i] == key[i]:
            count += 1
    accuracy = count/len(output_test)
    print 'test accuracy for NN = ', accuracy
    show_classifier(para, label, clf)
    



def KNN(para, label, nx, ny):
    X = para[:nx]
    y = label[:ny]
    clf = neighbors.KNeighborsClassifier(n_neighbors = 2)
    clf.fit(X,y)       
       
    output = clf.predict(para)
    count = 0.0
    for i in range(len(output)):
        if output[i] == label[i]:
            count += 1
    accuracy = count/len(output)
    print 'train accuracy for KNN = ', accuracy
    output_test = clf.predict(para[nx:])
    key = label[ny:]
    count = 0.0
    for i in range(len(output_test)):
        if output_test[i] == key[i]:
            count += 1
    accuracy = count/len(output_test)
    print 'test accuracy for KNN = ', accuracy
    show_classifier(para, label, clf)

def SVM(para, label, nx, ny):
    X = para[:nx]
    y = label[:ny]
    clf = svm.SVC()
    clf.fit(X,y)       
       
    output = clf.predict(para)
    count = 0.0
    for i in range(len(output)):
        if output[i] == label[i]:
            count += 1
    accuracy = count/len(output)
    print 'train accuracy for SVM = ', accuracy
    output_test = clf.predict(para[nx:])
    key = label[ny:]
    count = 0.0
    for i in range(len(output_test)):
        if output_test[i] == key[i]:
            count += 1
    accuracy = count/len(output_test)
    print 'test accuracy for SVM = ', accuracy
    show_classifier(para, label, clf)

def Tree(para, label, nx, ny):
    X = para[:nx]
    y = label[:ny]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,y)       
       
    output = clf.predict(para)
    count = 0.0
    for i in range(len(output)):
        if output[i] == label[i]:
            count += 1
    accuracy = count/len(output)
    print 'train accuracy for Decision Tree = ', accuracy
    output_test = clf.predict(para[nx:])
    key = label[ny:]
    count = 0.0
    for i in range(len(output_test)):
        if output_test[i] == key[i]:
            count += 1
    accuracy = count/len(output_test)
    print 'test accuracy for Decision Tree = ', accuracy
    show_classifier(para, label, clf)




def show_classifier(para, label, clf):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(1.0, 2.6, 0.001),
                         np.arange(-0.10, 0.25, 0.001))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter([para[i][0] for i in range(len(para))], [para[i][1] for i in range(len(para))], c = label, cmap = cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.show()
    '''
    Z = clf.predict_proba(para)
    print Z
    Z = np.array(Z)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    xx = np.arange(0, 2.5, 0.5)
    yy = np.arange(-0.15, 0.3, 0.05)
    ax = plt.plot()
    ax.contourf(xx,yy, Z, camp = cm, alpha = .8)
    ax.scatter(para[0], para[1], c = label, camp = cm_bright)
    '''
                  
                    
if __name__ == "__main__":

       #re_mouth = time_diff(group_remitters, 'EDWeek3')
       #non_mouth = time_diff(group_non, 'EDWeek3')
       #re_band = band_diff([3, 4, 5], [], 'EDWeek8', group_remitters)
       #non_band = band_diff([3, 4, 5], [], 'EDWeek8', group_non)
       #print len(re_band), re_band
       #print len(non_band), non_band
       #print stats.ttest_ind(re_band, non_band)
       #show_histogram(re_band, non_band)
       #re_FFBI = get_FFBI_diff('TEST_IAPS', group_remitters, 'EDWeek3', ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #non_FFBI = get_FFBI_diff('TEST_IAPS', group_non, 'EDWeek3', ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #re = get_para_FFBI('TEST_IAPS', group_remitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #non = get_para_FFBI('TEST_IAPS', group_non, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       re_iaps, non_iaps = get_IAPS_value_DC()
       #re_iaps, non_iaps = count_numbers_of_squares()
       print len(re_iaps), re_iaps
       print len(non_iaps), non_iaps
       print stats.ttest_ind(re_iaps, non_iaps)
       show_histogram(re_iaps, non_iaps)
       #print stats.ttest_ind(re_band, non_band)

       '''
       re = get_para_FFBI('TEST_IAPS', group_remitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       non = get_para_FFBI('TEST_IAPS', group_nonremitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       
       re_mouth = time_diff(group_remitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       non_mouth = time_diff(group_nonremitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       print sorted(re_mouth), sorted(non_mouth)
       #draw features
       pl.plot(re, re_mouth, 'ob', label = 'remitters')
       pl.plot(non, non_mouth, 'or', label = 'non-remitters')
       pl.legend()
       pl.show()
           
       
       
       re = get_para_FFBI('TEST_IAPS', group_remitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       renon = get_para_FFBI('TEST_IAPS', group_renon, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       non = get_para_FFBI('TEST_IAPS', group_non, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       
       re_mouth = time_diff(group_remitters, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       renon_mouth = time_diff(group_renon, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       non_mouth = time_diff(group_non, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       print sorted(re_mouth), sorted(non_mouth), sorted(renon_mouth)
       #draw features
       pl.plot(re, re_mouth, 'ob', label = 'remitters')
       pl.plot(renon, renon_mouth, 'oy', label = 'responder-nonremitters')
       pl.plot(non, non_mouth, 'or', label = 'non-responders')
       pl.legend()
       pl.show()
       '''
       '''
       re_FFBI = get_para_FFBI('TEST_IAPS', group_re, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       non_FFBI = get_para_FFBI('TEST_IAPS', group_non, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #print stats.ttest_ind(re_FFBI, non_FFBI)

       re_mouth = time_diff(group_re, ['EDWeek2_TGH', 'EDWeek2_TWH'])
       non_mouth = time_diff(group_non, ['EDWeek2_TGH', 'EDWeek2_TWH'])

       re_band = band_diff([3,4,5], [], 'EDWeek3', group_remitters)
       non_band = band_diff([3,4,5], [], 'EDWeek3', group_non)

       re_iaps_1, non_iaps_1 = get_IAPS_value_DC()
       re_iaps, non_iaps = count_numbers_of_squares()

       #re_FFBI = get_FFBI_diff('TEST_IAPS', group_remitters, 'EDWeek3', ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #non_FFBI = get_FFBI_diff('TEST_IAPS', group_non, 'EDWeek3', ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #print stats.ttest_ind(re_FFBI, non_FFBI)

       
       #get input and output
       para_FFBI = []
       para_Band = []
       para = []
       label = []
       re_FFBI_use = []
       re_Band_use = []
       non_FFBI_use = []
       non_Band_use = []
       for a, b in zip(re_iaps_1, re_iaps):
           if not math.isnan(a) and not math.isnan(b):
                para.append([a, b])
                para_FFBI.append(a)
                para_Band.append(b)
                label.append(1)
                re_FFBI_use.append(a)
                re_Band_use.append(b)

   
       for a, b in zip(non_iaps_1, non_iaps):
           if not math.isnan(a) and not math.isnan(b):
               para.append([a, b])
               para_FFBI.append(a)
               para_Band.append(b)
               label.append(0)
               non_FFBI_use.append(a)
               non_Band_use.append(b)

       #draw features
       pl.plot(re_FFBI_use, re_Band_use, 'ob', label = 'remitters')
       pl.plot(non_FFBI_use, non_Band_use, 'or', label = 'non-responders')
       pl.legend()
       #pl.show()



       index = range(len(para))
       random.shuffle(index)
       para = [para[i] for i in index]
       para_FFBI = [para_FFBI[i] for i in index]
       para_Mouth = [para_Band[i] for i in index]
       label = [label[i] for i in index]
       
       
#       NN(np.array(para_FFBI).reshape(len(para_FFBI), 1), label, 36, 36)
#       NN(np.array(para_Mouth).reshape(len(para_Mouth), 1), label, 36, 36)
       #NN(para, label, 30, 30)
       #KNN(para, label, 30, 30)
       #SVM(para, label, 30, 30)
       #Tree(para, label, 30, 30)
       plt.show()


       #re_FFWI = get_FFWI_diff('TEST_IAPS', group_remitters, 'EDWeek3', ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #non_FFWI = get_FFWI_diff('TEST_IAPS', group_non, 'EDWeek3', ['EDWeek2_TGH', 'EDWeek2_TWH'])
       #print stats.ttest_ind(re_FFWI, non_FFWI)
       '''