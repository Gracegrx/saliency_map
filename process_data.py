import pandas as pd
import numpy as np
from data_importer import DataImporter
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from PIL import Image
from utils import *
import csv
import os


group_non = ['4 D', '5 D', '6 D', '9 D', '11 D', '14 D', '15 D', '16 D', '17 D', '18 D', '23 D','24 D', '64 D', '70 D', '71 D', '72 D', '80 D', '81 D',
             '84 D', '88 D', '96 D']

group_remitters = ['1 D', '3 D', '7 D', '10 D', '12 D', '13 D', '22 D', '41 D', '47 D', '54 D', '68 D', '73 D', '75 D', '83 D', '85 D', '86 D', '93 D']

happy_face = ['F02HA', 'F01HA', 'M05HA', 'M10HA', 'F33HA', 'F30HA', 'M23HA', 'M31HA', 'F03HA', 'F26HA', 'M35HA',
              'M12HA', 'F16HA', 'F05HA', 'M06HA', 'M24HA', 'F31HA', 'F13HA', 'M13HA', \
              'M11HA', 'F14HA', 'F09HA', 'M22HA', 'M25HA', 'F07HA', 'F11HA', 'F17HA']
sad_face = ['F02SA', 'F01SA', 'M05SA', 'M10SA', 'F33SA', 'F30SA', 'M23SA', 'M31SA', 'F03SA', 'F26SA', 'M35SA', 'M12SA',
            'F16SA', 'F05SA', 'M06SA', 'M24SA', 'F31SA', 'F13SA', 'M13SA', \
            'M11SA', 'F14SA', 'F09SA', 'M22SA', 'M25SA', 'F07SA', 'F11SA', 'F17SA']

#for one image
def per_img_closure(fixation_list, subject, slide_no, presentation, slide_type, uniqueID, ImId, cat):
    duration_per_img = 0.0
    saliency_maps_per_img = np.zeros(DataImporter.saliency_map_size)
    for fix in fixation_list:
        saliency_map = fix.convert_to_saliency_map(DataImporter.saliency_map_size)
        saliency_maps_per_img += saliency_map
        duration_per_img += fix.dur

    output_array = [[subject, cat, presentation, slide_no, ImId, slide_type, uniqueID,
             saliency_maps_per_img, duration_per_img]]
    output_df = pd.DataFrame(output_array)
    #print output_df.columns
    return output_df

#need to be modified
def process_data(index=["Subject"], value=[group_remitters]):
    di = DataImporter(filename="../data/object_based_v2.txt")
    df = di.get_fix_sequence_df(per_img_closure,
                                 columns=["Subject", "cat", "Presentation", "Slide_no", "img_id", "slide_type", "UniqueID", "saliency_map", "duration"])
    for i in range(len(index)):
        df=df[df[index[i]].isin(value[i])]
    return df

def data_prepare():
    di = DataImporter(filename="../data/object_based_v2.txt")
    df = di.get_fix_sequence_df(per_img_closure,
                                columns=["Subject", "cat", "Presentation", "Slide_no", "img_id", "slide_type",
                                            "UniqueID", "saliency_map", "duration"])
    print "finish loading data!"
    return df

def data_filtrate(df, index, value):
    for i in range(len(index)):
        df = df[df[index[i]].isin(value[i])]
    return df


def extract_saliency(dataset):
    res = np.zeros(DataImporter.saliency_map_size)
    count = 0
    for (_, row) in dataset.iterrows():
        res += row["saliency_map"]
        count += 1

    result = res / count
    return result


def show_saliency(img, name):
    maxpix = np.amax(img)
    print "the max value of matrix is ", maxpix
    img = img*255/maxpix
    #fig = plt.figure()
    plt.imshow(img[:, :, 0])
    #fig.savefig(''+name+'.jpg')
    #plt.show()
    return img

def show_diff(img1, img2, name1, name2):
    res = img1 - img2
    maxpix = np.amax(res)
    minpix = np.amin(res)
    max = np.max(abs(maxpix), abs(minpix))
    res = res*128/max
    res = res + 128*np.ones(res.shape)
    fig = plt.figure()
    plt.imshow(res[:,:,0])
    #fig.savefig(''+name1+'_'+name2+'.jpg')
    #plt.show()
    return res


def single_image(df, index, value):
    name = ''
    res = data_filtrate(df, index, value)
    img = extract_saliency(res)
    #for i in range(len(index)):
    #    print value[i]
    #    name = name + index[i] + ' = ' + ''.join(value[i])
    img = show_saliency(img, name)
    #print res
    return img

def img_compare(df, index1, value1, index2, value2):

    res1 = data_filtrate(df, index1, value1)
    img1 = extract_saliency(res1)
    res2 = data_filtrate(df, index2, value2)
    img2 = extract_saliency(res2)
    name1 = ''+str(index1)+str(value1)
    #for i in range(len(index1)):
    #    name1 = name1 + index1[i] + ' = ' + ''.join(value1[i])
    name2 = ''+str(index2)+str(value2)
    #for i in range(len(index2)):
    #    name2 = name2 + index2[i] + ' = ' + ''.join(value2[i])
    res = show_diff(img1, img2, name1, name2)
    return res



def add_background(matrix, filename, savefilename):
    #img = imread(filename)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(matrix, cmap = 'gray')
    #fig.savefig('sal_'+savefilename)
    img = Image.open(filename).convert('L')
    img = np.array(img)
    img = imresize(img,(DataImporter.saliency_map_size[0],DataImporter.saliency_map_size[1]))
    res = img + matrix
    plt.subplot(1, 2, 2)
    plt.imshow(res, cmap = 'gray')
    fig.savefig(savefilename)
    #plt.show()


def iterate_results():
    result_dir = r"D:\Grace\pyRegionAnalyzer\object_based\result"
    bkgd_dir = r"D:\Grace\pyRegionAnalyzer\object_based\resources\img"

    df = data_prepare()
    filename = r"D:\Grace\pyRegionAnalyzer\object_based\data\object_based_v2.txt"
    file_data = pd.read_csv(filename, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in file_data.iterrows():
        img_pos = row[1]["ImId"]
        slide_num = row[1]["Slide_Num"]
        presentation = row[1]["Presentation"]

        if presentation != "ED Week 2_TWH":

            bkgd_file = "{}\{}\{}\{}.bmp".format(bkgd_dir, presentation, slide_num, img_pos)

            # for remitters
            save_dir = "{}\{}\{}\{}".format(result_dir, "remitters", presentation, slide_num)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            savefilename = "{}\{}.jpg".format(save_dir, img_pos)

            res = single_image(df, ["Subject", "Presentation", "Slide_no", "img_id"], [group_remitters, [str(presentation)], [str(slide_num)], [str(img_pos)]])
            np.savetxt('temp.txt', res)
            res = np.loadtxt('temp.txt', dtype='i', delimiter=' ')
            add_background(res, bkgd_file, savefilename)

            # for comparison
            save_dir = "{}\{}\{}\{}".format(result_dir, "non-responders", presentation, slide_num)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            savefilename = "{}\{}.jpg".format(save_dir, img_pos)

            res = single_image(df, ["Subject", "Presentation", "Slide_no", "img_id"], [group_non, [str(presentation)], [str(slide_num)], [str(img_pos)]])
            np.savetxt('temp.txt', res)
            res = np.loadtxt('temp.txt', dtype='i', delimiter=' ')
            add_background(res, bkgd_file, savefilename)


            #for comparison
            save_dir = "{}\{}\{}\{}".format(result_dir, "comparison", presentation, slide_num)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            savefilename = "{}\{}.jpg".format(save_dir, img_pos)

            res = img_compare(df, ["Subject", "Presentation", "Slide_no", "img_id"], [group_remitters, [str(presentation)], [str(slide_num)], [str(img_pos)]], ["Subject", "Presentation", "Slide_no", "img_id"], [group_non, [str(presentation)], [str(slide_num)], [str(img_pos)]])
            np.savetxt('temp.txt', res)
            res = np.loadtxt('temp.txt', dtype='i', delimiter=' ')
            add_background(res, bkgd_file, savefilename)



if __name__ == "__main__":

    #iterate_results()

    df = data_prepare()
    res = single_image(df, ["Subject", "UniqueID"],[group_remitters,["2058"]])
    #res = img_compare(df, ["Subject", "UniqueID"],[group_remitters,["2058"]],["Subject", "UniqueID"],[group_non,["2058"]])
    np.savetxt('temp.txt', res)

    res = np.loadtxt('temp.txt', dtype = 'i', delimiter = ' ')
    add_background(res, 'baby.jpg', 'test.jpg')

    #img1 = single_image(["UniqueID"],[["F01HA"]])


