import pandas as pd
import numpy as np
from data_importer import DataImporter
#from extract_images.get_extracted_images import GetExtractedImages
import matplotlib.pyplot as plt

group_non = ['4D', '5D', '6D', '9D', '11D', '14D', '15D', '16D', '17D', '18D', '23D','24D', '64D', '70D', '71D', '72D', '80D', '81D',
             '84D', '88D', '96D']

group_remitters = ['1D', '3D', '7D', '10D', '12D', '13D', '22D', '41D', '47D', '54D', '68D', '73D', '75D', '83D', '85D', '86D', '93D']

happy_face = ['F02HA', 'F01HA', 'M05HA', 'M10HA', 'F33HA', 'F30HA', 'M23HA', 'M31HA', 'F03HA', 'F26HA', 'M35HA',
              'M12HA', 'F16HA', 'F05HA', 'M06HA', 'M24HA', 'F31HA', 'F13HA', 'M13HA', \
              'M11HA', 'F14HA', 'F09HA', 'M22HA', 'M25HA', 'F07HA', 'F11HA', 'F17HA']
sad_face = ['F02SA', 'F01SA', 'M05SA', 'M10SA', 'F33SA', 'F30SA', 'M23SA', 'M31SA', 'F03SA', 'F26SA', 'M35SA', 'M12SA',
            'F16SA', 'F05SA', 'M06SA', 'M24SA', 'F31SA', 'F13SA', 'M13SA', \
            'M11SA', 'F14SA', 'F09SA', 'M22SA', 'M25SA', 'F07SA', 'F11SA', 'F17SA']

#for one image
def per_img_closure(fixation_list, subject, slide_no, presentation, slide_type, uniqueID, ImId):
    duration_per_img = 0.0
    saliency_maps_per_img = np.zeros(DataImporter.saliency_map_size)

    for fix in fixation_list:
        saliency_map = fix.convert_to_saliency_map(DataImporter.saliency_map_size)
        saliency_maps_per_img += saliency_map
        duration_per_img += fix.dur

    output_array = [[subject, presentation, slide_no, ImId, slide_type, uniqueID,
             saliency_maps_per_img, duration_per_img]]
    output_df = pd.DataFrame(output_array)
    #print output_df.columns
    return output_df

#need to be modified
def process_data(verbose=False):
    di = DataImporter(filename="../data/object_based_v2.txt")
    df = di.get_fix_sequence_df(per_img_closure,
                                 columns=["Subject", "Presentation", "Slide_no", "img_id", "slide_type", "UniqueID", "saliency_map", "duration"])
    #
    #print df
    filter_index = "Subject"
    filter_value = group_remitters
    #print(df[filter_index])
    filter = df[getattr(df, filter_index) in filter_value]
    print type(filter)
    print filter
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
    
if __name__ == "__main__":
    process_data()
