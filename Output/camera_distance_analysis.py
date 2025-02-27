import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image
# import sys
import os

real_img_root = './../testImages/images/'
real_label_root = './../testImages/labels/'

syn_img_root = 'train/images/'
syn_label_root ='train/labels/'

n_classes = 2

def get_size(files, root, class_to_test):
    x = []
    y = []
    for f in files:

        filename = os.path.join(root,f)

        # Open the file in read mode
        with open(filename, 'r') as file:
            for file_contents in file:
                if file_contents == '':
                    print('empty file')
                    continue

                class_id,x0,y0,w0,h0 = file_contents.split(' ')
                class_id = int(class_id)
                if class_id ==class_to_test:
                    w0 = float(w0)
                    h0 = float(h0)
                    # Print the contents of the file
                    x.append(w0)
                    y.append(h0)
    return np.array(x),np.array(y)



for class_to_test in range(n_classes):
    files = os.listdir(real_label_root)    
    x_real, y_real = get_size(files, real_label_root,class_to_test)
    files = os.listdir(syn_label_root)
    x_syn, y_syn = get_size(files, syn_label_root,class_to_test)


    bins = np.arange(x_real.min(), x_real.max(), 0.01)  
    # Plot the histogram with transparency for better visibility
    plt.hist(x_real, bins=bins, edgecolor='black', alpha=0.5, label='real', density=True)
    plt.hist(x_syn, bins=bins, edgecolor='black', alpha=0.5, label='synthetic',density = True)

    # Add labels and a legend
    plt.xlabel('Object width (pixels)')
    plt.ylabel('Frequency (normalized)')
    plt.legend()  # To show labels for the histograms
    plt.savefig(f'width_comp_{class_to_test}.png')
    plt.close()

    bins = np.arange(y_real.min(), y_real.max(), 0.01)  
    # Plot the histogram with transparency for better visibility
    plt.hist(y_real, bins=bins, edgecolor='black', alpha=0.5, label='real', density=True)
    plt.hist(y_syn, bins=bins, edgecolor='black', alpha=0.5, label='synthetic',density = True)

    # Add labels and a legend
    plt.xlabel('Object height (pixels)')
    plt.ylabel('Frequency (normalized)')
    plt.legend()  # To show labels for the histograms
    plt.savefig(f'height_comp_{class_to_test}.png')
    plt.close()

    width_ratio = x_syn.mean()/x_real.mean()
    height_ratio = y_syn.mean()/y_real.mean()
    print(f'Suggested width adjustment for class {class_to_test} = {width_ratio}')
    print(f'Suggested height adjustment for class {class_to_test} = {height_ratio}')
