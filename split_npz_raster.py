# Code to split a rasters .npz file obtained from concatenated recordings into individual raster.npy files
# Need to provide .npz file name AND file_list of original ch1 files (usually for Alejandro data)
# Output format is binarized data for each frame for each neuron: 
#    [[0,0,1,0,1,...]
#     [1,1,.........]
#      ...
#     [0,1,1,0......]]
import matplotlib.pyplot as plt
import numpy as np
import os

#Load original computed rasters
file_name = '/media/cat/250GB/in_vivo/alejandro/G2M4/joint/all_registered_processed_ROIs_deconvolved_data_thr2.0.npz'
data = np.load(file_name)
rasters = data['rasters']
traces = data['original_traces']
print (rasters.shape)
print rasters[0]


#Load filenames and lengths
all_names = np.loadtxt(os.path.split(file_name)[0]+'/all_names.txt', dtype='str')
all_lengths = np.loadtxt(os.path.split(file_name)[0]+'/all_lengths.txt')
print all_lengths
cumulative_lengths = []
ctr=0
cumulative_lengths.append(ctr)
for length in all_lengths:
    ctr+=length; cumulative_lengths.append(ctr)
    
print rasters[0]
cumulative_lengths = np.int32(cumulative_lengths)
print (cumulative_lengths)

#Convert rasters into milisecond precise time serios
rasters_binarized = np.zeros((len(rasters),int(np.sum(all_lengths))),dtype=np.int8)
for k in range(len(rasters)):
    rasters_binarized[k][np.int32(rasters[k])]=1
print rasters_binarized

#Split and save rasters back into their original folders
list_filename = '/media/cat/250GB/in_vivo/alejandro/G2M4/ch1_file_list.txt'       #********* Filenamelist
filenames = np.loadtxt(list_filename, dtype='str')
rasters_binarized_spontaneous = []
for s, filename in enumerate(filenames):
    ind1 = cumulative_lengths[s]
    ind2 = cumulative_lengths[s+1]
    print filename
    rasters_out = rasters_binarized[:,ind1:ind2]
    np.save(filename[:-4]+"_rasters", rasters_out)
quit() 
