# Code to compute louvain partitioning on a network
# Requires conversion of each file wise rasters; see split_npz_rasters.py file which takes wholetrack rasters and splits them
# Network edge weights defined by pair-wise correlation values 
# 
# TODO: -read more about Louvain partitioning; 
#       -there are other alternative methods to partition in addition to the louvain community module which re-use the networkx framework
#       -simplify correlation coefficicent computation

import community
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import xml.etree.ElementTree as ET

#This may be overkill --------------------------------- ALSO REDO USING SIMPLER CORRELATION COMPUTATIONS
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    
    
def compute_louvain_modularity(rasters):
    #Compute correlation matrices                   #TODO: Investigate other types of connectivity definitions
    corr_array = corr2_coeff(rasters, rasters)
    #print corr_array
    corr_array = np.nan_to_num(corr_array)
    #plt.imshow(corr_array)
    #plt.show()
    
    
    #Load non-negative matrix into a networkx graph
    G = corr_array-np.min(corr_array)
    G = nx.Graph(G)

    print "...# edges: ", G.number_of_edges()
    print "...# nodes: ", G.number_of_nodes()

    #first compute the best partition
    partition = community.best_partition(G)
    print len(partition.values())

    size = float(len(set(partition.values())))
    #print size
    pos = nx.spring_layout(G)
    count = 0.
    colors=np.array(['red','orange','yellow','green','cyan','blue','indigo','violet'])[::-1]
    plotting = False
    list_nodes_array = []
    for com in set(partition.values()) :
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]

        list_nodes_array.append(list_nodes)
        if plotting:
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 25+ np.exp(com),     #com/float(size)*1000,
                                    #node_color = str(count / size))
                                    #node_color = colors[com],alpha=.5)
                                    node_color = 'blue',alpha=.5)

    print list_nodes_array
    if plotting:
        nx.draw_networkx_edges(G, pos, alpha=0.25)
        plt.show()
    
    return list_nodes_array

   
    
#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
#G = nx.erdos_renyi_graph(100, 0.01)

#Load individual raster files and compute network matrix 
list_filename = '/media/cat/250GB/in_vivo/alejandro/G2M4/ch1_file_list.txt'
filenames = np.loadtxt(list_filename, dtype='str')
if False:
    for s, filename in enumerate(filenames):
        if '000' in filename:
            print ("... spontaneous recording ...")
            rasters = np.load(filename[:-4]+"_rasters.npy")
            list_nodes_array = compute_louvain_modularity(rasters)              #COMPUTES MODULARITY
            np.save(filename[:-4]+"_networks", list_nodes_array)

        if '001' in filename:
            print ("... stim recording... splitting ...")
            
            #Load rasters previously chunked
            rasters = np.load(filename[:-4]+"_rasters.npy")
            print rasters.shape
            
            #Load  times
            stim1_times = np.loadtxt(filename[:-4]+"_times_stim1.txt")
            stim2_times = np.loadtxt(filename[:-4]+"_times_stim2.txt")
            #print stim1_times
            
            #Load frame rate for recording:
            tree = ET.parse(filename[:-8]+'.xml')  
            root = tree.getroot()
            if root[0][3].attrib['key'] == 'framePeriod':
                inter_frame = root[0][3].attrib['value'] #['framePeriod']
                frame_rate = 1./float(inter_frame)
                print ('...frame rate: ', frame_rate)
            else:
                print ("... can't find frame rate... exiting...")
                quit()
            
            #Convert stim times to frame times:
            stim1_times_frame_numbers = np.int32(stim1_times * 1E-3 *frame_rate)
            stim2_times_frame_numbers = np.int32(stim2_times * 1E-3 *frame_rate)
     
            #Make rasters triggered off stim times + 2seconds 
            rasters_stim1=[]
            rasters_stim2=[]
            offset = 2 * frame_rate #period of activity to load into matrices
            for stim1,stim2 in zip(stim1_times_frame_numbers,stim2_times_frame_numbers):
                #print stim1, stim2
                rasters_stim1.append(rasters[:,stim1:int(stim1+offset)])
                rasters_stim2.append(rasters[:,stim2:int(stim2+offset)])
            
            rasters_stim1 = np.hstack(rasters_stim1)
            np.save(filename[:-4]+"_rasters_stim1", rasters_stim1)
            rasters_stim2 = np.hstack(rasters_stim2)
            np.save(filename[:-4]+"_rasters_stim2", rasters_stim2)
            
            #Compute modularity values and save
            if True: 
                modularity_stim1 = compute_louvain_modularity(rasters_stim1)    #COMPUTES MODULARITY
                np.save(filename[:-4]+"_networks_stim1", modularity_stim1)
                modularity_stim2 = compute_louvain_modularity(rasters_stim2)    #COMPUTES MODULARITY
                np.save(filename[:-4]+"_networks_stim2", modularity_stim2)
    quit()

#Visualize network outputs
#Load ROI countour data first
roi_filename = '/media/cat/250GB/in_vivo/alejandro/G2M4/joint/all_registered_processed_ROIs.npz'
data_in = np.load(roi_filename, encoding= 'latin1',  mmap_mode='c')

Bmat_array = data_in['Bmat_array']
cm = data_in['cm']                      #Centre of mass    
thr_array = data_in['thr_array']        #Threshold array original data, usually 0.2
traces = data_in['traces']              #
x_array = data_in['x_array']
y_array = data_in['y_array']
colors='b'

thr_fixed=.5
ctr=0
modularity_levels = np.arange(0,25,1)
colors = ['gold','mediumslateblue','grey','thistle','teal','palegreen','violet','deepskyblue','blue','green','cyan','orange','red']
for s, filename in enumerate(filenames):
    print (filename)
    
    if '000' in filename:
        print ctr, (ctr*7)%21+int(ctr/3.)+1
        print ("... spontaneous recording ...")
        network_spont = np.load(filename[:-4]+"_networks.npy")
        
        ax=plt.subplot(3,7,(ctr*7)%21+int(ctr/3.)+1)
        ax.set_xticks([]); ax.set_yticks([])
        if ctr==0: 
            plt.ylabel("Spontaneous",fontsize=15)
        ctr+=1
        
        plt.title(os.path.split(filename)[1][:-4].replace('_C1V1_GCaMP6s','')+", #: "+str(len(network_spont)),fontsize=9)

        #Draw all neurons first
        for i, (y,x,Bmat,thr) in enumerate(zip(y_array,x_array,Bmat_array,thr_array)):
            cs = plt.contour(y, x, Bmat, [thr_fixed], colors='black',alpha=0.3)

        #Draw neurons at each modularity 
        for k in modularity_levels:
            if k>(len(network_spont)-1): break
            index_array = network_spont[k]      #Select neurons at this level

            unique_indexes=np.unique(index_array) #THIS is redundant as neurons are uniquely asigned to modularity levels
            print ("... modularity: ", k, " # neurons: ", len(unique_indexes))
            for i in unique_indexes:
                cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors=colors[k],linewidth=15, alpha=1)

        #plt.show()
        
    if '001' in filename:

        print ("... stim recording...")

        #Draw stim1 ensembles
        network_stim1 = np.load(filename[:-4]+"_networks_stim1.npy")
        ax=plt.subplot(3,7,(ctr*7)%21+int(ctr/3.)+1)
        ax.set_xticks([]); ax.set_yticks([])
        if ctr==1:
            plt.ylabel("Horizontal",fontsize=15)
        ctr+=1
        plt.title(os.path.split(filename)[1][:-4].replace('_C1V1_GCaMP6s','')+", #: "+str(len(network_stim1)),fontsize=9)

        #Draw all neurons first
        for i, (y,x,Bmat,thr) in enumerate(zip(y_array,x_array,Bmat_array,thr_array)):
            cs = plt.contour(y, x, Bmat, [thr_fixed], colors='black',alpha=0.3)

        #Draw neurons at each modularity 
        for k in modularity_levels:
            if k>(len(network_stim1)-1): break
            index_array = network_stim1[k]      #Select neurons at this level

            unique_indexes=np.unique(index_array) #THIS is redundant as neurons are uniquely asigned to modularity levels
            print ("... modularity: ", k, " # neurons: ", len(unique_indexes))
            for i in unique_indexes:
                cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors=colors[k],linewidth=15, alpha=1)
        
        
        #Draw stim2 ensembles
        network_stim2 = np.load(filename[:-4]+"_networks_stim2.npy")
        ax=plt.subplot(3,7,(ctr*7)%21+int(ctr/3.)+1)
        ax.set_xticks([]); ax.set_yticks([])
        if ctr==2:
            plt.ylabel("Vertical",fontsize=15)
        ctr+=1
        plt.title(os.path.split(filename)[1][:-4].replace('_C1V1_GCaMP6s','')+", #: "+str(len(network_stim2)),fontsize=9)

        #Draw all neurons first
        for i, (y,x,Bmat,thr) in enumerate(zip(y_array,x_array,Bmat_array,thr_array)):
            cs = plt.contour(y, x, Bmat, [thr_fixed], colors='black',alpha=0.3)

        #Draw neurons at each modularity 
        for k in modularity_levels:
            if k>(len(network_stim2)-1): break
            index_array = network_stim2[k]      #Select neurons at this level

            unique_indexes=np.unique(index_array) #THIS is redundant as neurons are uniquely asigned to modularity levels
            print ("... modularity: ", k, " # neurons: ", len(unique_indexes))
            for i in unique_indexes:
                cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors=colors[k],linewidth=15, alpha=1)
        
        


plt.show()

    















