import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from Fixations import FixationsList

class DataImporter:
    saliency_map_size = (128, 128, 1)
    def __init__(self, filename):
        self.file_data = pd.read_csv(filename, delimiter="\t", quoting=csv.QUOTE_NONE)
               
    def get_fix_sequence_df(self, closure, columns):
        output_array = []
        for (subject, cat, slide_no, uniqueID, img_pos, presentation, slide_type,
             fixation_list) in self._iterate_file_data(self.file_data):
            output = closure(fixation_list, subject, slide_no, presentation, slide_type, uniqueID, img_pos, cat)
            output_array.append(output)
        output_df = pd.concat(output_array)
        #print "columns of output_df is: ", output_df.columns
        output_df.columns = columns
        return output_df
        
    def _iterate_file_data(self, file_data):
        subject_data = {}
        for row in file_data.iterrows():
            fix_x = self._split_by_comma(row[1]["FixationPositionsX"])
            temp_length = len(fix_x)
            fix_y = self._split_by_comma(row[1]["FixationPositionsY"])
            fix_dur = self._split_by_comma(row[1]["FixationDurations_ms"])
            fix_start = self._split_by_comma(row[1]["FixationStart"])
            fix_loc_x = []
            fix_loc_y = []
            fix_size_x = []
            fix_size_y = []
            i = 0
            while (i<temp_length):
                fix_loc_x.append(row[1]["Loc_x"])
                fix_loc_y.append(row[1]["Loc_y"])
                fix_size_x.append(row[1]["Size_x"])
                fix_size_y.append(row[1]["Size_y"])
                i += 1


            fix_list = FixationsList.from_pos(
                fix_x, fix_y, fix_start, fix_dur, fix_loc_x, fix_loc_y, fix_size_x, fix_size_y)
            #print "length of fixationlist = ", len(fix_list)

            uniqueID = row[1]["uniqueImgID(s)"]
            img_pos = row[1]["ImId"]
            subject = row[1]["Subject"]
            slide_num = row[1]["Slide_Num"]
            slide_type = row[1]["slideType(s)"]
            presentation = row[1]["Presentation"]
            cat = row[1]["cat"]
            if subject not in subject_data:
                subject_data[subject] = {}

            identifier = (uniqueID, img_pos, presentation, slide_num, slide_type, cat)
            if identifier not in subject_data[subject]:
                subject_data[subject][identifier] = fix_list
            else:
                subject_data[subject][identifier] = subject_data[subject][identifier] + fix_list
                
        for subject_i in subject_data:
            for (uniqueID, img_pos, presentation, slide_num, slide_type, cat) in subject_data[subject_i]:
                yield subject_i, cat, slide_num, uniqueID, img_pos, presentation, slide_type, subject_data[subject_i][(uniqueID, img_pos,
                    presentation, slide_num, slide_type, cat)]
            
    def _split_by_comma(self, comma_string):
        output = []
        array = comma_string.split(",")
        for i in array:
            i_o = i.replace("\"", "")
            if self.is_float(i_o):
                output.append(float(i_o))
        return output
                
    def is_float(self, s):
        try: 
            float(s)
            return True
        except ValueError:
            return False
