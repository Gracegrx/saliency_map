from PIL import Image
import pandas as pd
import os

input_path = r"D:\jon\eye tracking\Data\Alzheimer\Part 2\Images"
input_description = r"D:\jon\eye tracking\Data\Alzheimer\Part 2\Images\OutputFile.txt"
output_path = r"D:\jon\analysis\Alzheimer\cut_images\Part2"

def get_coords_from_description():
	coords = {}
	df = pd.read_csv(input_description)
	counter = 0
	for row in df.iterrows():
		row_content = row[1][0]

		if "\t" in row_content and row_content[0].isdigit():
			row_array = row_content.split("\t")
			x = int(row_array[1])
			y = int(row_array[2])
			w = int(row_array[3])
			h = int(row_array[4])
			
			slide_num = counter / 4
			image_num = counter % 4
			coords[(slide_num + 1, image_num + 1)] = (x, y, x+w, y+h)
			# {"x": x, "y": y, "w": w, "h": h}
			counter += 1
	return coords

def cut_image_from_coords(slide_name, coord):
	img = Image.open(input_path + "\\" + slide_name)
	img_crop = img.crop(coord)
	return img_crop
	
def main():
	coords = get_coords_from_description()
	for k, v in coords.iteritems():
		slide_name = str(k[0]) + ".bmp" 
		slide_directory = output_path + "\\" + str(k[0])
		image_name = str(k[1]) + ".bmp"
		cropped_img = cut_image_from_coords(slide_name, v)
		
		if not os.path.exists(slide_directory):
			os.makedirs(slide_directory)
		
		cropped_img.save(slide_directory + "\\" + image_name)
		
if __name__ == "__main__":
	main()