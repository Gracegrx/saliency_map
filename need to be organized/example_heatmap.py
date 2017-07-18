import matplotlib
matplotlib.use('Qt4Agg')

from Heatmap.heatmap import Heatmap
from Heatmap.Heatmap_iterator import Heatmap_Iterator
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


#different groups of patients
group_re = ['Subject1D', 'Subject3D', 'Subject7D', 'Subject8D', 'Subject10D', 'Subject12D', 'Subject13D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject22D', 'Subject25D', \
                'Subject41D', 'Subject47D', 'Subject54D', 'Subject68D', 'Subject73D', 'Subject75D', 'Subject83D', 'Subject85D', 'Subject86D', 'Subject90D', 'Subject93D']

#abandon 'Subject2D'
#group_non = ['Subject2D']
group_non = ['Subject4D','Subject5D', 'Subject6D', 'Subject9D', 'Subject11D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject23D', \
                'Subject24D', 'Subject64D', 'Subject70D', 'Subject71D', 'Subject72D', 'Subject80D', 'Subject81D', 'Subject84D', 'Subject88D', 'Subject96D']

group_depress = ['Subject1D', 'Subject3D', 'Subject7D', 'Subject8D', 'Subject10D', 'Subject12D', 'Subject13D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject22D', 'Subject25D', \
                'Subject41D', 'Subject47D', 'Subject54D', 'Subject68D', 'Subject73D', 'Subject75D', 'Subject83D', 'Subject85D', 'Subject86D', 'Subject90D', 'Subject93D', 'Subject2D', 'Subject4D','Subject5D', 'Subject6D', 'Subject9D', 'Subject11D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject23D', \
                'Subject24D', 'Subject64D', 'Subject70D', 'Subject71D', 'Subject72D', 'Subject80D', 'Subject81D', 'Subject84D', 'Subject88D', 'Subject96D']
                
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
                    
group_remitters = ['Subject1D', 'Subject3D', 'Subject7D', 'Subject10D', 'Subject12D', 'Subject13D', 'Subject22D', \
                'Subject41D', 'Subject47D', 'Subject54D', 'Subject68D', 'Subject73D', 'Subject75D', 'Subject83D', 'Subject85D', 'Subject86D', 'Subject93D']
                
group_renon = ['Subject8D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject25D', 'Subject90D']

group_nonremitters = ['Subject2D', 'Subject4D','Subject5D', 'Subject6D', 'Subject9D', 'Subject11D', 'Subject14D', 'Subject15D', 'Subject16D', 'Subject17D', 'Subject18D', 'Subject23D', \
                'Subject24D', 'Subject64D', 'Subject70D', 'Subject71D', 'Subject72D', 'Subject80D', 'Subject81D', 'Subject84D', 'Subject88D', 'Subject96D', 'Subject8D', 'Subject19D', 'Subject20D', 'Subject21D', 'Subject25D', 'Subject90D']

unique_ids = ['1121', '1661', '7220','7289','9342','9468','9471','9472','9417','9831','9926','9927','2235','2311','2312','2339','2055.2','2056','2151','2392', \
                '2058','8461','2141','3350','8531','8540','8600','8620','3301','2216','2276','2304','4700','9421','4610','3300','2540','9415','9419', \
              '2360','8501','8502','8503','8510','2900.1','2301','2260','2045','8400','8490','8497','8499','2057','9041','9000','8120','2340','4599','2205','2520', \
              '8460','8465','8467','8470','8200','2345','9210','2750','5593','5594','5600','5629','2208','2530','9560','9530','4614','9220','2075','9332','2190','2191', \
              '2200','2206','2155','2160','2170','2222','2071','3220','9190','2303','2240','2250','2270','2274','2217','2272','2273','2278','4641','9001','2456','2165', \
              '2302','2306','2308','2342','9435','1340','2703','2299','2300','2305','2396','2397','2359','2362','2370','2373','2455','3230','2341','2501','2210','2214','2215',\
              '2221','2224','2095','2154','9331','5470','5301','5890','5450']
all_unique_ids = ['1121', '1661', '7220','7289','9342','9468','9471','9472','9417','9831','9926','9927','2235','2311','2312','2339','2055.2','2056','2151','2392', \
                '2058','8461','2141','3350','8531','8540','8600','8620','3301','2216','2276','2304','4700','9421','4610','3300','2540','9415','9419', \
              '2360','8501','8502','8503','8510','2900.1','2301','2260','2045','8400','8490','8497','8499','2057','9041','9000','8120','2340','4599','2205','2520', \
              '8460','8465','8467','8470','8200','2345','9210','2750','5593','5594','5600','5629','2208','2530','9560','9530','4614','9220','2075','9332','2190','2191', \
              '2200','2206','2155','2160','2170','2222','2071','3220','9190','2303','2240','2250','2270','2274','2217','2272','2273','2278','4641','9001','2456','2165', \
              '2302','2306','2308','2342','9435','1340','2703','2299','2300','2305','2396','2397','2359','2362','2370','2373','2455','3230','2341','2501','2210','2214','2215',\
              '2221','2224','2095','2154','9331','5470','5301','5890','5450', 'F02HA', 'F01HA', 'M05HA', 'M10HA', 'F33HA', 'F30HA', 'M23HA', 'M31HA', 'F03HA', 'F26HA', 'M35HA', 'M12HA', 'F16HA', 'F05HA', 'M06HA', 'M24HA', 'F31HA', 'F13HA', 'M13HA', \
                    'M11HA', 'F14HA', 'F09HA', 'M22HA', 'M25HA', 'F07HA', 'F11HA', 'F17HA', 'F02SA', 'F01SA', 'M05SA', 'M10SA', 'F33SA', 'F30SA', 'M23SA', 'M31SA', 'F03SA', 'F26SA', 'M35SA', 'M12SA', 'F16SA', 'F05SA', 'M06SA', 'M24SA', 'F31SA', 'F13SA', 'M13SA', \
                    'M11SA', 'F14SA', 'F09SA', 'M22SA', 'M25SA', 'F07SA', 'F11SA', 'F17SA']

weeks = ['EDWeek2_TGH', 'EDWeek3', 'EDWeek4', 'EDWeek5', 'EDWeek6', 'EDWeek7', 'EDWeek8']

mouth = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]

#choose a group of patients and a week
#calculate the normalized time difference between happy and sad faces within the region.(value between 0 and 1)
#return an array of values, each subject has a value.
def time_diff(group, week, region):
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
        value.append(temp1.get_region(region)*rhappy-temp2.get_region(region)*rsad)
    value = np.array(value)
    value = value[~np.isnan(value)]
#    print value
    return value

def diff_subject(subject, week, region):
    temp1 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=week, Subject=subject, uniqueID=happy_face, m=10, n=10)
    temp2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=week, Subject=subject, uniqueID=sad_face, m=10, n=10)
    value = temp1.get_region(region) - temp2.get_region(region)
    return value


##choose a group of patients and a week
#calculate the exact time difference between happy and sad faces within the region.(regardless of the total time difference)
#return an array of values, each subject has a value.
def diff_group(group, week, region):
    value = []
    for subject in group:
        value.append(diff_subject(subject, week, region))
    value = np.array(value)
    value = value[~np.isnan(value)]
    print value
    return value

#input are two lists.
#draw a histogram with two labels.
def show_histogram(list1, list2, label1='remitters', label2='non-responders'):
    #bins = np.arange(min(min(list1), min(list2))-0.05, max(max(list1), max(list2))+0.05, 0.02)
    bins = 'auto'
    plt.xlim([min(min(list1), min(list2))-0.05, max(max(list1), max(list2))+0.05])
    plt.hist(list1, bins = bins, alpha = 0.3, label = label1)
    plt.hist(list2, bins = bins, alpha = 0.3, label = label2)
    plt.title(label1+' vs '+label2)
    plt.legend(loc='upper right')
    plt.show()

#compare difference from week2 to week8 for a single subject.
#indicate a region to be concerned.
#draw a pic to show the changes through different weeks.
def person(subject, region = mouth):
    change = []
    for week in weeks:
        change.append(diff_subject(subject, week, region))
    print change
    x = [2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, change)
    plt.show()


#n1 indicates the index of bands that we'd like to take into account
#n2 indicates the index of bands that we'd like to kick out
#returns an array, one value for each subject.
def band_distribution(n1, n2, week, group):
    re_value = []
    for i in range(len(group)):
        res = 0
        week2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group[i], m=1, n=10)
        week3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=week, Subject=group[i], m=1, n=10)
        for num in n1:
            res += week3.get_diff(week2)[num]
        for num in n2:
            res -= week3.get_diff(week2)[num]
        re_value.append(res)
    re_value = np.array(re_value)
    re_value = re_value[~np.isnan(re_value)]
    return re_value

'''
def band_distribution_time(n1, n2, week, group):
    re_value = []
    for i in range(len(group)):
        res = 0
        week2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group[i], m=1, n=10)
        week3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=week, Subject=group[i], m=1, n=10)
        week2_time += week2.fixation_time()
        week3_time += week3.fixation_time()
        for num in n1:
            res += week3.get_diff(week2)[num]
        for num in n2:
            res -= week3.get_diff(week2)[num]
        re_value.append(res)
    re_value = np.array(re_value)
    re_value = re_value[~np.isnan(re_value)]
    return re_value
'''

#decide which part of region should be considered later
#slide_num is just for final show_result, can be ignored if we just need the value
def region_of_interest(slide_num, unique_id, x = 590, y = 462):
    ctrl = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_ctrl, ori_img='resources/week2/'+slide_num+'.bmp', size_x=x, size_y=y, uniqueID=unique_id, m=20, n=20)
    depress = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_depress, ori_img='resources/week2/'+slide_num+'.bmp', size_x=x, size_y=y, uniqueID=unique_id, m=20, n=20)
    emotional = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=[group_remitted, group_ctrl], ori_img='resources/week2/'+slide_num+'.bmp', size_x=x, size_y=y, uniqueID=unique_id, m=20, n=20)
    matrix_emo = ctrl.get_ori_size_matrix()
    matrix = depress.get_diff(ctrl)
    #print ctrl.get_ori_size_matrix()
    #print matrix
    #hs = ctrl.load()
    #if len(hs) != 0:
    #    ctrl.show_roi_grey(matrix_emo, 0.01)
    #return matrix
    return matrix_emo
    #depress.show_compare(ctrl)

def show_rois_iaps(slide_num, unique_id, boundary, x = 590, y = 462):
    ctrl = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_ctrl,
                            ori_img='resources/week2/' + slide_num + '.bmp', size_x=x, size_y=y, uniqueID=unique_id,
                            m=20, n=20)
    matrix_emo = ctrl.get_ori_size_matrix()
    hs = ctrl.load()
    if len(hs) != 0:
        ctrl.show_roi_grey(matrix_emo, boundary)
    return matrix_emo


def get_fixation_value(matrix, slide_num, unique_id, subject, x = 590, y = 462):
    week2 = Heatmap_Iterator(Slide_type = 'TEST_IAPS', Week = 'EDWeek2_TGH', Subject = subject, ori_img='resources/week2/'+slide_num+'.bmp', size_x = x, size_y = y, uniqueID = unique_id, m=20, n=20)
    week3 = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek3', Subject=subject, ori_img='resources/week3/' + slide_num + '.bmp', size_x=x, size_y=y, uniqueID=unique_id, m=20, n=20)
    subtract = week3.get_diff(week2)
    res = 0.0
    for i in range(len(subtract)):
        for j in range(len(subtract[i])):
            if matrix[i][j] > 0:
            #if matrix[i][j] > 0.01:
                res += subtract[i][j]
            elif matrix[i][j] < 0:
                res += subtract[i][j]
            else:
                res = res
    #print res
    return res

def count_roi(matrix, slide_num, unique_id, subject, x = 590, y = 462):
    week2 = Heatmap_Iterator(Slide_type = 'TEST_KDEF', Week = 'EDWeek2_TGH', Subject = subject, ori_img='resources/week2/'+slide_num+'.bmp', size_x = x, size_y = y, uniqueID = unique_id, m=10, n=10)
    week3 = Heatmap_Iterator(Slide_type='TEST_KDEF', Week='EDWeek3', Subject=subject, ori_img='resources/week3/' + slide_num + '.bmp', size_x=x, size_y=y, uniqueID=unique_id, m=10, n=10)
    #week2 = Heatmap_Iterator(Slide_type = ['TEST_IAPS', 'TEST_KDEF'], Week = 'EDWeek2_TGH', Subject = subject, ori_img='resources/week2/'+slide_num+'.bmp', size_x = x, size_y = y, uniqueID = unique_id, m=10, n=10)
    #week3 = Heatmap_Iterator(Slide_type = ['TEST_IAPS', 'TEST_KDEF'], Week='EDWeek3', Subject=subject, ori_img='resources/week3/' + slide_num + '.bmp', size_x=x, size_y=y, uniqueID=unique_id, m=10, n=10)

    subtract = week3.get_diff(week2)
    res = np.zeros(2)
    for i in range(len(subtract)):
        for j in range(len(subtract[i])):
            if matrix[i][j] >= 0:
            #if matrix[i][j] > 0.01:
                if subtract[i][j] > 0:
                    res[0] += 1
                elif subtract[i][j] < 0:
                    res[1] += 1
                else:
                    res = res
            else:
                res = res
    #print res
    return res



ids = {'8':['2058', '8461', '2141', '3350'],'11':['3301', '2216', '2276', '2304'], '13':['4700', '9421', '4610', '3300'], \
       '14': ['2540', '9415', '9419', '2360'],'17':['2900.1', '2301', '2260', '2045'],'21':['2057', '9041', '9000', '8120'], \
       '22': ['2340', '4599', '2205', '2520'],'25':['8200', '2345', '9210', '2750'],'29':['2208', '2530', '9560', '9530'], \
       '31': ['4614', '9220', '2075', '9332'],'34': ['2071', '3220', '9190', '2303'],'38': ['4641', '9001', '2456', '2165'], \
       '42': ['9435', '1340', '2703', '2299'],'46': ['2455', '3230', '2341', '2501'],'48': ['2224', '2095', '2154', '9331'], \
       }

all_ids = {'8':['2058', '8461', '2141', '3350'],'11':['3301', '2216', '2276', '2304'], '13':['4700', '9421', '4610', '3300'], \
       '14': ['2540', '9415', '9419', '2360'],'17':['2900.1', '2301', '2260', '2045'],'21':['2057', '9041', '9000', '8120'], \
       '22': ['2340', '4599', '2205', '2520'],'25':['8200', '2345', '9210', '2750'],'29':['2208', '2530', '9560', '9530'], \
       '31': ['4614', '9220', '2075', '9332'],'34': ['2071', '3220', '9190', '2303'],'38': ['4641', '9001', '2456', '2165'], \
       '42': ['9435', '1340', '2703', '2299'],'46': ['2455', '3230', '2341', '2501'],'48': ['2224', '2095', '2154', '9331'], \
        '10': ['F02HA', 'F01HA', 'M05HA', 'M10HA', 'F33HA', 'F30HA', 'M23HA', 'M31HA', 'F03HA', 'F26HA', 'M35HA', 'M12HA', 'F16HA', 'F05HA', 'M06HA', 'M24HA', 'F31HA', 'F13HA', 'M13HA', \
                    'M11HA', 'F14HA', 'F09HA', 'M22HA', 'M25HA', 'F07HA', 'F11HA', 'F17HA', 'F02SA', 'F01SA', 'M05SA', 'M10SA', 'F33SA', 'F30SA', 'M23SA', 'M31SA', 'F03SA', 'F26SA', 'M35SA', 'M12SA', 'F16SA', 'F05SA', 'M06SA', 'M24SA', 'F31SA', 'F13SA', 'M13SA', \
                    'M11SA', 'F14SA', 'F09SA', 'M22SA', 'M25SA', 'F07SA', 'F11SA', 'F17SA']}


def look_into_heatmaps():
    # for num in ids:
    num = '8'
    if num in ids:
        for id in ids[num]:
            # idd = '2058'
            # slidenum = '8'
            example1_re = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_remitters,
                                           ori_img='resources/week2/' + num + '.bmp', size_x=590, size_y=442,
                                           uniqueID=id, m=20, n=20)
            example1_re.show_result()
            example2_re = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek3', Subject=group_remitters,
                                           ori_img='resources/week3/' + num + '.bmp', size_x=590, size_y=442,
                                           uniqueID=id, m=20, n=20)
            # example2_re.show_result()
            example1_non = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_non,
                                            ori_img='resources/week2/' + num + '.bmp', size_x=590, size_y=442,
                                            uniqueID=id, m=20, n=20)
            # example1_non.show_result()
            example2_non = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek3', Subject=group_non,
                                            ori_img='resources/week3/' + num + '.bmp', size_x=590, size_y=442,
                                            uniqueID=id, m=20, n=20)
            # example2_non.show_result()
            example1_ctrl = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_ctrl,
                                             ori_img='resources/week2/' + num + '.bmp', size_x=590, size_y=442,
                                             uniqueID=id, m=20, n=20)
            #example1_ctrl.show_result()
            example1_depress = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_depress,
                                                ori_img='resources/week2/' + num + '.bmp', size_x=590, size_y=442,
                                                uniqueID=id, m=20, n=20)
            # example1_depress.show_result()
            #example2_re.show_compare(example1_re)
            #example2_non.show_compare(example1_non)
            example1_depress.show_compare(example1_ctrl)

def calculate_roi_values():

    remitters = np.zeros(len(group_remitters))
    non_responders = np.zeros(len(group_non))
    for id in unique_ids:
        matrix = region_of_interest('13', id)
        #print matrix
        for i in range(len(group_remitters)):
            temp = get_fixation_value(matrix, '13', id, group_remitters[i])
            #count = count_roi(matrix, '13', id, group_remitters[i])
            if temp != 'nan':
                #print temp
                remitters[i] += temp
        for i in range(len(group_non)):
            temp = get_fixation_value(matrix, '13', id, group_non[i])
            if temp != 'nan':
                non_responders[i] += temp
    remitters = np.array(remitters)
    non_responders = np.array(non_responders)
    remitters = remitters[~np.isnan(remitters)]
    non_responders = non_responders[~np.isnan(non_responders)]
    print len(remitters), remitters
    print len(non_responders), non_responders
    print stats.ttest_ind(remitters, non_responders)

def show_rois():
    #for num in ids:
        num = '8'
        for id in ids[num]:
            show_rois_iaps(num, id, 0.01)

def count_numbers_of_squares():
    remitters = np.zeros(shape=(len(group_remitters), 2))
    non_responders = np.zeros(shape=(len(group_non), 2))
    res_re = []
    res_non = []
    for id in all_unique_ids:
        #matrix = region_of_interest('13', id)
        matrix = np.zeros(shape = (10,10))
        for i in range(len(group_remitters)):
            count = count_roi(matrix, '13', id, group_remitters[i])
            remitters[i][0] += count[0]
            remitters[i][1] += count[1]
        for i in range(len(group_non)):
            count = count_roi(matrix, '13', id, group_non[i])
            non_responders[i][0] += count[0]
            non_responders[i][1] += count[1]
    for i in range(len(remitters)):
        temp = remitters[i][0] * 1.0 / remitters[i][1]
        temp = np.array(temp)
        temp = temp[~np.isnan(temp)]
        res_re.append(temp)
    for i in range(len(non_responders)):
        temp = non_responders[i][0] * 1.0 / non_responders[i][1]
        temp = np.array(temp)
        temp = temp[~np.isnan(temp)]
        #print non_responders[i][0], non_responders[i][1]
        res_non.append(temp)

    # remitters = np.array(remitters)
    # non_responders = np.array(non_responders)
    # remitters = remitters[~np.isnan(remitters)]
    # non_responders = non_responders[~np.isnan(non_responders)]
    print len(res_re), res_re
    print len(res_non), res_non
    print stats.ttest_ind(res_re, res_non)
    return res_re, res_non


def count_numbers_of_squares_ANOVA():
    re = []
    non = []
    for id in all_unique_ids:
        remitters = np.zeros(shape=(len(group_remitters), 2))
        non_responders = np.zeros(shape=(len(group_non), 2))
        res_re = []
        res_non = []
        #matrix = region_of_interest('13', id)
        matrix = np.zeros(shape = (10,10))
        for i in range(len(group_remitters)):
            count = count_roi(matrix, '13', id, group_remitters[i])
            remitters[i][0] += count[0]
            remitters[i][1] += count[1]
        for i in range(len(group_non)):
            count = count_roi(matrix, '13', id, group_non[i])
            non_responders[i][0] += count[0]
            non_responders[i][1] += count[1]
        for i in range(len(remitters)):
            temp = remitters[i][0] * 1.0 / remitters[i][1]
            temp = np.array(temp)
            temp = temp[~np.isnan(temp)]
            temp = temp.tolist()
            res_re.extend(temp)

        print res_re
        re.append(res_re)
        for i in range(len(non_responders)):
            temp = non_responders[i][0] * 1.0 / non_responders[i][1]
            temp = np.array(temp)
            temp = temp[~np.isnan(temp)]
            # print non_responders[i][0], non_responders[i][1]
            temp = temp.tolist()
            res_non.extend(temp)
        print res_non
        non.append(res_non)
    print len(re), len(non)
    print len(re[0]), len(non[0])
    print re
    print non

    # remitters = np.array(remitters)
    # non_responders = np.array(non_responders)
    # remitters = remitters[~np.isnan(remitters)]
    # non_responders = non_responders[~np.isnan(non_responders)]
    #print len(res_re), res_re
    #print len(res_non), res_non
    #print stats.ttest_ind(res_re, res_non)
    return res_re, res_non

if __name__ == "__main__":
    #look_into_heatmaps()
    #count_numbers_of_squares()
    count_numbers_of_squares_ANOVA()
    #show_rois()
    #look_into_heatmaps()
    #a = Heatmap_Iterator('TEST_KDEF','EDWeek2_TGH',Subject = group_non)
    #a.show_result()

    #re = band_distribution([3,4,5], [], 'EDWeek3', group_remitters)
    #non = band_distribution([3,4,5], [], 'EDWeek3', group_non)
    #print stats.ttest_ind(re, non)
    '''
    #count numbers of red and blue squares
    remitters = np.zeros(shape = (len(group_remitters), 2))
    non_responders = np.zeros(shape = (len(group_non), 2))
    res_re = []
    res_non = []
    for id in unique_ids:
        matrix = region_of_interest('13', id)
        for i in range(len(group_remitters)):
            count = count_roi(matrix, '13', id, group_remitters[i])
            remitters[i][0] += count[0]
            remitters[i][1] += count[1]
        for i in range(len(group_non)):
            count = count_roi(matrix, '13', id, group_non[i])
            non_responders[i][0] += count[0]
            non_responders[i][1] += count[1]
    for i in range(len(remitters)):
        res_re.append(remitters[i][0]*1.0/remitters[i][1])
    for i in range(len(non_responders)):
        res_non.append(non_responders[i][0]*1.0/non_responders[i][1])

    # remitters = np.array(remitters)
    # non_responders = np.array(non_responders)
    # remitters = remitters[~np.isnan(remitters)]
    # non_responders = non_responders[~np.isnan(non_responders)]
    print len(res_re), res_re
    print len(res_non), res_non
    print stats.ttest_ind(res_re, res_non)
    '''

    '''
    for num in ids:
        for id in ids[num]:
            show_rois_iaps(num, id, 0.01)


    remitters = np.zeros(len(group_remitters))
    non_responders = np.zeros(len(group_non))
    for id in unique_ids:
        matrix = region_of_interest('13', id)
        #print matrix
        for i in range(len(group_remitters)):
            temp = get_fixation_value(matrix, '13', id, group_remitters[i])
            #count = count_roi(matrix, '13', id, group_remitters[i])
            if temp != 'nan':
                #print temp
                remitters[i] += temp
        for i in range(len(group_non)):
            temp = get_fixation_value(matrix, '13', id, group_non[i])
            if temp != 'nan':
                non_responders[i] += temp
    remitters = np.array(remitters)
    non_responders = np.array(non_responders)
    remitters = remitters[~np.isnan(remitters)]
    non_responders = non_responders[~np.isnan(non_responders)]
    print len(remitters), remitters
    print len(non_responders), non_responders
    print stats.ttest_ind(remitters, non_responders)
    '''
    '''
    #re = band_distribution([3,4,5], [], 'EDWeek3', group_remitters)
    #non = band_distribution([3,4,5], [], 'EDWeek3', group_non)
    #print len(re), len(non)
    #show_histogram(re, non)

    #for num in ids:
    num = '25'
    if num in ids:
        for id in ids[num]:
            #idd = '2058'
            #slidenum = '8'
            example1_re = Heatmap_Iterator(Slide_type = 'TEST_IAPS', Week = 'EDWeek2_TGH', Subject = group_remitters, ori_img = 'resources/week2/'+num+'.bmp', size_x = 590, size_y = 442, uniqueID = id, m=20, n=20)
            example1_re.show_result()
            example2_re = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek3', Subject=group_remitters, ori_img='resources/week3/'+num+'.bmp', size_x=590, size_y=442, uniqueID=id, m=20, n=20)
            #example2_re.show_result()
            example1_non = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_non, ori_img='resources/week2/'+num+'.bmp', size_x=590, size_y=442, uniqueID=id, m=20, n=20)
            #example1_non.show_result()
            example2_non = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek3', Subject=group_non, ori_img='resources/week3/'+num+'.bmp', size_x=590, size_y=442, uniqueID=id, m=20, n=20)
            #example2_non.show_result()
            example1_ctrl = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_ctrl, ori_img='resources/week2/'+num+'.bmp', size_x=590, size_y=442, uniqueID=id, m=20, n=20)
            #example1_ctrl.show_result()
            example1_depress = Heatmap_Iterator(Slide_type='TEST_IAPS', Week='EDWeek2_TGH', Subject=group_depress, ori_img='resources/week2/'+num+'.bmp', size_x=590, size_y=442, uniqueID=id, m=20, n=20)
            #example1_depress.show_result()
            example2_re.show_compare(example1_re)
            example2_non.show_compare(example1_non)
            #example1_depress.show_compare(example1_ctrl)
    '''
    '''
    #alist = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject='Subject4D', uniqueID = sad_face)
    re_sad_2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_remitters, uniqueID = sad_face, m = 10, n = 10)
    re_happy_2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_remitters, uniqueID = happy_face, m = 10, n = 10)
    re_sad_3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek8'], Subject=group_remitters, uniqueID=sad_face, m = 10, n = 10)
    re_happy_3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek8'], Subject=group_remitters, uniqueID=happy_face, m = 10, n = 10)
    non_sad_2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_non, uniqueID=sad_face, m = 10, n = 10)
    non_happy_2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_non, uniqueID=happy_face, m = 10, n = 10)
    non_sad_3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek8'], Subject=group_non, uniqueID=sad_face, m = 10, n = 10)
    non_happy_3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek8'], Subject=group_non, uniqueID=happy_face, m = 10, n = 10)
    '''
    '''
    week2 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_remitters, m = 1, n = 10)
    week3 = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek5'], Subject=group_remitters, m = 1, n = 10)
    week2_non = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek2_TGH'], Subject=group_non, m = 1, n = 10)
    week3_non = Heatmap_Iterator(Slide_type=['TEST_KDEF'], Week=['EDWeek5'], Subject=group_non, m = 1, n = 10)
    res = week3.get_diff(week2)
    week2.show_compare_matrix(res)
    res_non = week3_non.get_diff(week2_non)
    week2_non.show_compare_matrix(res_non)
    '''
    '''
    re_happy = re_happy_3.get_diff(re_happy_2)
    re_sad = re_sad_3.get_diff(re_sad_2)
    non_happy = non_happy_3.get_diff(non_happy_2)
    non_sad = non_sad_3.get_diff(non_sad_2)
    #Heatmap_Iterator.show_compare(blist, alist)
    #matrix = alist.get_diff(blist)
    #print re_happy, re_sad, non_happy, non_sad
    result_re = re_happy+re_sad
    result_non = non_happy + non_sad
    print result_re
    print result_non
    re_sad_2.show_compare_matrix(result_re)
    non_sad_2.show_compare_matrix(result_non)
    #alist.show_result()
    #alist.show_result_color()
    '''


    '''
#     person('Subject73D')   
    
    
#    re = time_diff(group_re, 'EDWeek2_TGH', mouth)
#    non = time_diff(group_non, 'EDWeek2_TGH', mouth)
#    print stats.ttest_ind(re, non)
#    show_histogram(re, non, 'Week8_re', 'Week8_non')
    
    remitters_sad = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_remitters, uniqueID = sad_face)
    remitters = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_remitters)
    ctrl = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_ctrl)
    bi = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_bi)
    nonres = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_non)
    nonres_sad = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_non, uniqueID = sad_face)
    remitters_happy = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_remitters, uniqueID = happy_face)
    nonres_happy = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_non, uniqueID = happy_face)
    
#    remitters_sad.show_result()
#    remitters_happy.show_result()
#    nonres_sad.show_result()
#    nonres_happy.show_result()    
#    Heatmap_Iterator.show_compare(remitters_happy, nonres_happy)
#    Heatmap_Iterator.show_compare(remitters_sad, nonres_sad)
    remitters.show_result()  
    nonres.show_result()
    ctrl.show_result()
    bi.show_result()
    Heatmap_Iterator.show_compare(remitters, nonres)
    '''
    
    '''
    #hypothesis 2:    
    matrix = dlist.get_ori_size_matrix()
    re_value= []
    for i in range(len(group_re)):
        temp1 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_re[i], uniqueID = happy_face, m = 10, n = 10)
        temp2 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_re[i], uniqueID = sad_face, m = 10, n = 10)
        re_value.append(temp1.decide(matrix))
    re_value = np.array(re_value)
    re_value = re_value[~np.isnan(re_value)]
    print re_value
    
    non_value = []
    for i in range(len(group_non)):
        temp1 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_non[i], uniqueID = happy_face, m = 10, n = 10)
        temp2 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_non[i], uniqueID = sad_face, m = 10, n = 10)
        non_value.append(temp1.decide(matrix))
    non_value = np.array(non_value)
    non_value = non_value[~np.isnan(non_value)]
    print non_value
    
    
    
    depress_value = []
    for i in range(len(group_depress)):
        temp1 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_depress[i], uniqueID = happy_face, m = 10, n = 10)
        temp2 = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_depress[i], uniqueID = sad_face, m = 10, n = 10)
        depress_value.append(temp1.decide(matrix))
    depress_value = np.array(depress_value)
    depress_value = depress_value[~np.isnan(depress_value)]
    print depress_value
    
    print 'ttest result for responders and non_responders):\n', stats.ttest_ind(re_value, non_value)
    print 'ttest result for responders and whole depressed people):\n', stats.ttest_ind(re_value, depress_value)
    print 'ttest result for non_responders and whole depressed people):\n', stats.ttest_ind(non_value, depress_value)    
    
  
    alist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_non, uniqueID = sad_face, m = 10, n = 10)
#    print alist.get_pro_matrix()
#    blist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_non, uniqueID = happy_face, m = 10, n = 10)
#    clist = Heatmap_Iterator(Slide_type = ['TEST_IAPS'], Week = ['EDWeek3'], Subject = ['Subject10D'], ori_img = 'resources/22_week3.bmp', size_x = 590, size_y = 442, ImId = 'ImId4', m = 10, n = 10)
#    dlist = Heatmap_Iterator(Slide_type = ['TEST_IAPS'], Week = ['EDWeek8'], Subject = ['Subject10D'], ori_img = 'resources/22_week8.bmp', size_x = 590, size_y = 442, ImId = 'ImId2', m = 10, n = 10)
#    alist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH', 'EDWeek2_TWH'], Subject = group_uni, ori_img = 'resources/10_week2_ori.bmp', m = 10, n = 10)
#    blist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TWH', 'EDWeek2_TGH'], Subject = group_bi, ori_img = 'resources/10_week2_bi.bmp', size_x = 310, size_y = 410, m = 10, n = 10)

#    blist.show_result()
      
#    elist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_re, uniqueID = sad_face, m = 10, n = 10)      
#    flist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_re, uniqueID = happy_face, m = 10, n = 10)  
    clist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_ctrl, uniqueID = happy_face, m = 10, n = 10)
#    clist.show_result()
#    print alist.get_pro_matrix()
    dlist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_ctrl, uniqueID = sad_face, m = 10, n = 10)
    elist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_undepress, uniqueID = happy_face, m = 10, n = 10)
    flist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_undepress, uniqueID = sad_face, m = 10, n = 10)
    glist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_remitted, uniqueID = happy_face, m = 10, n = 10)
    hlist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_remitted, uniqueID = sad_face, m = 10, n = 10)
#    Heatmap_Iterator.show_compare(elist, clist)    
#    matrix = Heatmap_Iterator.get_diff(clist, dlist)
#    print matrix
#    alist.show_roi(matrix)
#    Heatmap_Iterator.show_compare(clist, elist)
#    print Heatmap_Iterator.get_diff(blist, clist)
    
#    print dlist.decide(matrix)-clist.decide(matrix)    
#    print blist.decide(matrix)-alist.decide(matrix)
#    print flist.decide(matrix)-elist.decide(matrix)
#get a list for responders

    
    
    elist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_re, m = 10, n = 10)
    flist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_re, m = 10, n = 10)
    Heatmap_Iterator.show_compare(elist, flist)
    print Heatmap_Iterator.get_diff(elist, flist)
    glist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek2_TGH'], Subject = group_non, m = 10, n = 10)
    hlist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_non, m = 10, n = 10)
    Heatmap_Iterator.show_compare(glist, hlist)
    print Heatmap_Iterator.get_diff(glist, hlist)
    
#    clist.show_result()
#    dlist.show_result()
#    clist.show_result()
#    print clist.get_pro_matrix()
#    dlist.show_result()
#    blist.show_result()
#    print Heatmap_Iterator.compare(alist, blist)
#    print show_result(alist.load())
#    show_result(blist.load())

    clist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_re, m = 10, n = 10)
    dlist = Heatmap_Iterator(Slide_type = ['TEST_KDEF'], Week = ['EDWeek8'], Subject = group_non, m = 10, n = 10)
    Heatmap_Iterator.show_compare(clist, dlist)
    print Heatmap_Iterator.get_diff(clist, dlist)
    clist.show_result()
    dlist.show_result()

    
#    alist = Heatmap_Iterator(Slide_type = ['TEST_IAPS'], Week = ['EDWeek1'], ori_img = 'resources/8.bmp', m = 10, n = 10, uniqueID = ['3350'], size_x = 590, size_y = 442)
    alist = Heatmap_Iterator(Slide_type = ['TEST_IAPS'], Week = ['EDWeek1'], ori_img = 'resources/8.bmp', m = 10, n = 10, uniqueID = ['8461'], size_x = 590, size_y = 442)
    alist.show_result()
   
   
    
'''    

