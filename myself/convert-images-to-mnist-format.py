
#neeeeeeeeed rebuild in windows
import os
from PIL import Image
from array import *
from random import shuffle

# Load from and save to
dataset_root = r'C:\Users\user\Desktop\CapsNet-Tensorflow-myself\myself'
Names = [[dataset_root+r'\training-images','train'], [dataset_root+r'\test-images','test']]

for name in Names:
	print('---new name---')
	print('name', name)
	data_image = array('B')
	data_label = array('B')

	FileList = []
	for dirname in os.listdir(name[0]): #cancel [1:] because need all labels
		print('---new dirname---')
		print('dirname', dirname)
		path = os.path.join(name[0], dirname)
		print('path', path)
		for filename in os.listdir(path):
			if filename.endswith(".png"):
				FileList.append(os.path.join(name[0],dirname,filename))

	shuffle(FileList) # Usefull for further segmenting the validation set 對於進一步細分驗證集很有用

	for filename in FileList:
		label = int(filename.split('\\')[-2])

		Im = Image.open(filename)
		pixel = Im.load()
		width, height = Im.size

		for x in range(0,width):
			for y in range(0,height):
				grayscale = (pixel[y,x][0]+pixel[y,x][1]+pixel[y,x][2])//3
				data_image.append(grayscale) #here use RGB means to black-white

		data_label.append(label) # labels start (one unsigned byte each)

	hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
	print('hexval:', hexval)

	# header for label array
	header = array('B')
	header.extend([0,0,8,1,0,0])
	header.append(int('0x'+hexval[2:][:2],16))
	header.append(int('0x'+hexval[2:][2:],16))
	print('header:', header)

	data_label = header + data_label

	# additional header for images array
	if max([width,height]) <= 256:
		header.extend([0,0,0,width,0,0,0,height])
	else:
		raise ValueError('Image exceeds maximum size: 256x256 pixels');

	header[3] = 3 # Changing MSB for image data (0x00000803)
	data_image = header + data_image

	print('output_file:', dataset_root+'\\'+name[1]+'-images-idx3-ubyte')
	output_file = open(dataset_root+'\\'+name[1]+'-images-idx3-ubyte', 'wb')
	data_image.tofile(output_file)
	output_file.close()

	print('output_file:', dataset_root+'\\'+name[1]+'-labels-idx1-ubyte')
	output_file = open(dataset_root+'\\'+name[1]+'-labels-idx1-ubyte', 'wb')
	data_label.tofile(output_file)
	output_file.close()

# gzip resulting files

#for name in Names:
#	os.system('gzip '+name[1]+'-images-idx3-ubyte')
#	os.system('gzip '+name[1]+'-labels-idx1-ubyte')