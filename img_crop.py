from PIL import Image
import pandas as pd
import os
from process_data import *
import csv

dir_input = r"D:\Grace\pyRegionAnalyzer\object_based\resources\slides"
dir_output = r"D:\Grace\pyRegionAnalyzer\object_based\resources\img"


def crop_img(area, filename, savefilename):
    img = Image.open(filename)
    cropped_img = img.crop(area)
    #cropped_img.show()
    cropped_img.save(savefilename)


def iterate_images():
    filename = r"D:\Grace\pyRegionAnalyzer\object_based\data\object_based_v2.txt"
    file_data = pd.read_csv(filename, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in file_data.iterrows():
        fix_loc_x = row[1]["Loc_x"]
        fix_loc_y = row[1]["Loc_y"]
        fix_size_x = row[1]["Size_x"]
        fix_size_y = row[1]["Size_y"]
        img_pos = row[1]["ImId"]
        slide_num = row[1]["Slide_Num"]
        presentation = row[1]["Presentation"]

        if presentation != "ED Week 2_TWH":
            area = (int(fix_loc_x),int(fix_loc_y),int(fix_loc_x)+ int(fix_size_x), int(fix_loc_y)+ int(fix_size_y))
            save_dir = "{}\{}\{}".format(dir_output,presentation, slide_num)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = "{}\{}\{}.bmp".format(dir_input, presentation, slide_num)
            savefilename = "{}\{}.bmp".format(save_dir, img_pos)
            crop_img(area, filename, savefilename)

if __name__ == "__main__":

    iterate_images()