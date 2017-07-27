import pandas as pd
import numpy as np
from data_importer import DataImporter
#from extract_images.get_extracted_images import GetExtractedImages
import matplotlib.pyplot as plt
from PIL import Image

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
    return df

def data_filtrate(df, index, value):
    for i in range(len(index)):
        df = df[df[index[i]].isin(value[i])]
    return df


def show_saliency(dataset, name):
    res = np.zeros(DataImporter.saliency_map_size)
    count = 0
    for(_, row) in dataset.iterrows():
        res += row["saliency_map"]
        count += 1

    result = res/count
    maxpix = np.amax(result)
    print "the max value of matrix is ", maxpix
    result = result*255/maxpix
    #img = Image.fromarray(np.uint(result[:, :, 0]), 'L')
    #img.save("image_test.jpg")
    fig = plt.figure()
    plt.imshow(result[:, :, 0])
    fig.savefig(''+name+'.jpg')
    plt.show()
    return result

def show_saliency_diff(img1, img2, name1, name2):
    res = (img1-img2)[:,:,0]
    rangee = np.amax(res) - np.amin(res)
    temp_matrix = (255 / rangee) * np.ones(res.shape)
    matrix = np.amin(res) * np.ones(res.shape)
    adjust_diff = res - matrix
    adjust_diff = adjust_diff * temp_matrix
    img = Image.fromarray(255 * np.uint8(adjust_diff), 'RGB')
    maximum = np.amax(res)
    minimum = np.amin(res)
    for i in range(len(adjust_diff)):
        for j in range(len(adjust_diff[i])):
            if adjust_diff_ori[i][j] >= 0:
                num = min(255, int(255 * (1 - (maximum - res[i][j]) / maximum)))
                img.putpixel((j, i), (num, 0, 0))
            else:
                num = min(255, int(255 * (1 - (minimum - res[i][j]) / minimum)))
                img.putpixel((j, i), (0, 0, num))
                #        img.show()
    plt.imshow(img)
    fig = plt.figure()
    fig.savefig('compare'+name1+name2+'.jpg')
    return img

def show_diff_from_img(img1, img2, name1, name2):
    #matrix1 =
    return 0


def add_background(saliency_map, image):



    return 0

    '''
    hs = self.load()
    region = hs[0].get_region()
    im = Image.open(self.ori_img)
    img1 = im.crop(region)
    out = ImageChops.multiply(img1, img)
    out.show()
    img = Image.fromarray(np.uint(res[:, :, 0]), 'L')
    '''

    '''
    for(_, row) in filter.iterrows():
        subject = row["Subject"]
        presentation = row["Presentation"]
        slide_no = row["Slide_no"]
        img_no = row["img_id"]
        img_type = row["img_type"]
        if verbose:
            print("Processing: {}:{}-{}".format(subject, presentation, slide_no))
        first_saliency = row["saliency_map"]
        first_duration = row["duration"]
    '''

    '''
    startdf = df[df["slide_type"] == "start"]
    pairs = []
    for (_, row) in startdf.iterrows():
        subject = row["Subject"]
        presentation = row["Presentation"]
        slide_no = row["Slide_no"]
        img_no = row["img_no"]
        img_type = row["img_type"]
        if verbose:
            print("Processing: {}:{}-{}".format(subject, presentation, slide_no))
        first_saliency = row["saliency_map"]
        first_duration = row["duration"]

        if img_type == "1-back":
            repeated_slide_no = slide_no + 1
        else:
            repeated_slide_no = slide_no + 2

        df_index_arr = [df["Subject"] == subject, df["Presentation"] == presentation,
                        df["Slide_no"] == repeated_slide_no, df["img_no"] == img_no]
        df_index = np.ones(shape=(df.shape[0],))
        for i in df_index_arr:
            df_index = np.logical_and(df_index, i)
        pair_extraction = df[df_index]
        if pair_extraction.shape[0] != 1:
            if verbose:
                print ("No pair found")
            continue
        assert pair_extraction.shape[0] > 0, "More than one image extracted"
        pair_extraction = pair_extraction.iloc[0]

        pair_presentation = pair_extraction["Presentation"]
        pair_slide_no = pair_extraction["Slide_no"]
        pair_img_no = pair_extraction["img_no"]
        repeated_saliency = pair_extraction["saliency_map"]
        repeated_duration = pair_extraction["duration"]

        image1 = GetExtractedImages.by(presentation, slide_no, img_no)
        image2 = GetExtractedImages.by(pair_presentation, pair_slide_no, pair_img_no)
        if np.array_equal(image1, image2) == False:
            print subject, presentation, slide_no
            plt.subplot(2, 1, 1)
            plt.imshow(image1)
            plt.subplot(2, 1, 2)
            plt.imshow(image2)
            plt.show()
        assert np.array_equal(image1, image2), "Image not equal"
        pairs.append([subject, presentation, slide_no, repeated_slide_no, img_no,
                      image1, first_saliency, repeated_saliency, first_duration,
                      repeated_duration])
    output_df = pd.DataFrame(pairs, columns=["Subject", "Presentation", "Slide_no",
                                             "Repeated_slide_no", "Img_no", "Image",
                                             "First_saliency", "Repeated_saliency",
                                             "First_duration", "Repeated_duration"])
    # output_df.to_csv("data/extracted_data1.csv")
    return output_df
    '''

def single_image(index, value):
    name = ''
    df = data_prepare()
    res = data_filtrate(df, index, value)
    for i in range(len(index)):
        name = name + index[i] + ' = ' + ''.join(value[i])
    show_saliency(res, name)
    #print res
    return [res, name]


if __name__ == "__main__":

    img1 = single_image(["UniqueID"],[["F01HA"]])
    print type(img1[0]), type(img1[1])
    print img1[0], img1[1]
    #img1 = single_image(["cat", "slide_type"],[["BD"], ["TEST_KDEF"]])
    #img2 = single_image(["cat", "slide_type"],[["D","R","C"], ["TEST_KDEF"]])
    #show_saliency_diff(img1[0], img2[0], img1[1], img2[1])

