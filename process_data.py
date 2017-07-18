import pandas as pd
import numpy as np
from vsb_data_manager.data_importer import DataImporter
from extract_images.get_extracted_images import GetExtractedImages
import matplotlib.pyplot as plt

#for one image
def per_img_closure(fixation_list, subject, slide_no, presentation, slide_type, uniqueID, ImId):
    duration_per_img = 0.0
    saliency_maps_per_img = np.zeros(DataImporter.saliency_map_size)

    for fix in fixation_list:
        img_pos = fix.img_pos
        uniqueID = fix.uniqueID
        saliency_map = fix.convert_to_saliency_map(DataImporter.saliency_map_size)
        saliency_maps_per_img += saliency_map
        uniqueID_per_img[img_pos] = uniqueID
        duration_per_img[img_pos] += fix.dur

    output_array = [subject, presentation, slide_no, ImId, slide_type, uniqueID,
             saliency_maps_per_img, duration_per_img]
    output_df = pd.DataFrame(output_array)
    return output_df

#need to be modified
def process_data(verbose=False):
    di = DataImporter(filename="data/object_based_v1.txt")
    df = di.get_fix_sequence_df(per_img_closure,
                                 columns=["Subject", "Presentation", "Slide_no", "img_no",
                                          "slide_type", "img_type", "saliency_map", "duration"])
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
    
if __name__ == "__main__":
    process_data()
