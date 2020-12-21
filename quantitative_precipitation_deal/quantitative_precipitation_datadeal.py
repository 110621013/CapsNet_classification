import os
from PIL import Image
import numpy as np
#from skimage.util import img_as_float

# this .py is used for quantitative_precipitation_deal data deal
tr_te = ['training_images', 'testing_images']
zero_one = ['0', '1']
train_num, test_num = 719, 349
img_size = 28

def make_data():
    # train or test
    for tt in tr_te:
        if tt == 'training_images':
            tt_array = np.full((train_num, img_size, img_size), -1, dtype=np.uint8)
            tt_array_label = np.full((train_num), -1, dtype=np.int8)
        elif tt == 'testing_images':
            tt_array = np.full((test_num, img_size, img_size), -1, dtype=np.uint8)
            tt_array_label = np.full((test_num), -1, dtype=np.int8)
        else:print('tt error!!!')

        print(tt_array.shape, tt_array_label.shape)

        count = 0
        # one or zero
        for zo in zero_one:
            for dirPath, dirNames, fileNames in os.walk(os.path.join(os.getcwd(), "quantitative_precipitation_deal", tt, zo)):
                print(dirPath)
                for i in range(len(fileNames)):
                    # im_array shape:28*28*3, value:0~255(numpy.uint8)
                    im_array = np.array(Image.open(os.path.join(dirPath, fileNames[i])), dtype=np.uint8)

                    # RGB averger, singal_im_array shape:28*28
                    singal_im_array = np.full((img_size, img_size), -1, dtype=np.uint8)
                    for i in range(img_size):
                        for j in range(img_size):
                            singal_im_array[i, j] = im_array[i, j, 0]/3 + im_array[i, j, 1]/3 + im_array[i, j, 2]/3

                    tt_array[count] = singal_im_array
                    tt_array_label[count] = int(zo)
                    count += 1

        '''#####
        for i in range(5):
            print('label:', tt_array_label[i], tt_array_label[-i])
            img = Image.fromarray(np.uint8(tt_array[i]), 'L')
            img.show()
            img = Image.fromarray(np.uint8(tt_array[-i]), 'L')
            img.show()
        #####'''

        # randomly shuffle
        rng = np.random.default_rng()
        seed = []
        for i in range(tt_array_label.shape[0]):seed.append(i)
        rng.shuffle(seed)
        print(seed[0:5])

        tt_array_shuffled = np.empty_like(tt_array)
        tt_array_label_shuffled = np.empty_like(tt_array_label)
        for i in range(tt_array_label.shape[0]):
            tt_array_shuffled[i] = tt_array[seed[i]]
            tt_array_label_shuffled[i] = tt_array_label[seed[i]]

        '''#####
        for i in range(5):
            print('label:', tt_array_label_shuffled[i], tt_array_label_shuffled[-i])
            img = Image.fromarray(np.uint8(tt_array_shuffled[i]), 'L')
            img.show()
            img = Image.fromarray(np.uint8(tt_array_shuffled[-i]), 'L')
            img.show()
        #####'''
        print(tt_array_shuffled[0])

        np.save(os.path.join(os.getcwd(), 'data', 'quantitative_precipitation', tt.split('_')[0]+'_data2'), tt_array_shuffled)
        np.save(os.path.join(os.getcwd(), 'data', 'quantitative_precipitation', tt.split('_')[0]+'_label2'), tt_array_label_shuffled)

if __name__ == "__main__":
    print(type(np.load(os.path.join(os.getcwd(), 'data', 'quantitative_precipitation', 'training_data2.npy'))[0]))
    print(type(np.load(os.path.join(os.getcwd(), 'data', 'quantitative_precipitation', 'training_label2.npy'))[0]))
    print(type(np.load(os.path.join(os.getcwd(), 'data', 'quantitative_precipitation', 'testing_data2.npy'))[0]))
    print(type(np.load(os.path.join(os.getcwd(), 'data', 'quantitative_precipitation', 'testing_label2.npy'))[0]))