from PIL import Image
img = Image.open('3.bmp')
area = (144,41,446,441)
cropped_img = img.crop(area)
cropped_img.show()
cropped_img.save('bkgd.jpg')