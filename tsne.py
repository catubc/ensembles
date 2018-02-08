# Kang Miller et al 2014 method for computing ensembles  
# 
# 
# 


import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

def PCA(X, n_components):
    from sklearn import decomposition

    #pca = decomposition.SparsePCA(n_components=3, n_jobs=1)
    pca = decomposition.PCA(n_components=n_components)

    print "... fitting PCA ..."
    pca.fit(X)
    
    for k in range (n_components):
        print "... explained variance: ", pca.explained_variance_[k]

    print "... pca transform..."
    return pca.transform(X)
        
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    
def jaeun_detect(rasters, list_filename):
    
    print rasters.shape
    
    activity = np.sum(rasters,axis=0)
    #plt.plot(activity)

    std = np.std(activity)*4
    #plt.plot([0,len(rasters[0])],[std,std])

    #determine neurons in each ensemble:
    indexes = np.where(activity>std)[0]
    ensembles = []
    vectors = []
    for index in indexes:
        ensembles.append(np.where(rasters[:,index]>0)[0])
        print len(ensembles)
        vectors.append(rasters[:,index])
        #plt.plot([index,index],[0,40])
    #plt.show()
    
    #Normalize vectors
    vec_matrix = np.float32(np.vstack(vectors)).T
    print vec_matrix.shape
    for k in range(len(vec_matrix)):
        vec_matrix[k] = vec_matrix[k]/np.sum(vec_matrix[k])
    vec_matrix = np.nan_to_num(vec_matrix).T
    
    plt.imshow(vec_matrix)
    plt.show()
   
    #Compute correlation:
    corr_matrix = np.zeros((len(vectors),len(vectors)),dtype=np.float32)
    corr_array =[]
    for e in range(len(vectors)): 
        for f in range(0,len(vectors),1):
            print len(vectors[e]),len(vectors[f])
            print scipy.stats.pearsonr(vectors[e],vectors[f])
            r = scipy.stats.pearsonr(vectors[e],vectors[f])[1]
            corr_matrix[e,f] = 0.5*np.log((1+r)/(1-r))
            corr_array.append(0.5*np.log((1+r)/(1-r)))
            
    print corr_matrix
    print np.min(corr_matrix), np.max(corr_matrix)
    plt.imshow(corr_matrix)
    plt.show()
    
    #from scipy.cluster.hierarchy import linkage, dendrogram
    #linkage_matrix = linkage(vectors, 'single')
    #dendogram = dendrogram(linkage_matrix,truncate_mode='none')
    from sklearn.cluster import SpectralClustering
    mat = corr_matrix
    ids = SpectralClustering(10).fit_predict(mat)
    #print img    
    
    
    if False: #Use PCA to cluster highly-active frames:
        data = PCA(vectors,3)
        plt.scatter(data[:,0], data[:,1])
        plt.show()
        
    if False:       #Use hyperangles to compute difference 
        matrix = np.zeros((len(vectors),len(vectors)), dtype=np.float32)
        vector0 = np.zeros(len(vectors[0]),dtype=np.float32)+1
        dists = []
        for e in range(len(vectors)):
            p = np.dot(vectors[e],vector0)/np.linalg.norm(vectors[e])/np.linalg.norm(vector0) # -> cosine of the angle
            dists.append(np.degrees(np.arccos(np.clip(p, -1, 1))))

            for f in range(len(vectors)):
                c = np.dot(vectors[e],vectors[f])/np.linalg.norm(vectors[e])/np.linalg.norm(vectors[f]) # -> cosine of the angle
                #dists = 
                #print c
                #c = np.degrees(np.arccos(np.clip(c, -1, 1)))
                matrix[e,f] = np.degrees(np.arccos(np.clip(c, -1, 1)))
                #matrix[e,f] = c
                print "...angle: ", matrix[e,f]
                #temp_angle_array.append(c)

        plt.imshow(matrix)
        plt.show()
        
        bin_width = 10   # histogram bin width in usec
        
        y = np.histogram(dists, bins = np.arange(0,90,bin_width))
        plt.bar(y[1][:-1], y[0], bin_width, color='b', alpha=1)
        plt.show()
    
    if False:   #Use SVD on all data;
        pass
        
    
    #quit()
    
    return ensembles, ids


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


list_filename = '/media/cat/250GB/in_vivo/alejandro/G2M4/ch1_file_list.txt'
filenames = np.loadtxt(list_filename, dtype='str')

thr_fixed=.5
ctr=0
modularity_levels = np.arange(0,25,1)
colors = ['gold','mediumslateblue','grey','thistle','teal','palegreen','violet','deepskyblue','blue','green','cyan','orange','red']
for s, filename in enumerate(filenames):
    print (filename)
    
    if '000' in filename:
        print ctr, (ctr*7)%21+int(ctr/3.)+1
        print ("... spontaneous recording ...")
        rasters = np.load(filename[:-4]+"_rasters.npy")
        
        #ensembles, ids = jaeun_detect(rasters, list_filename)
        frame_array, weight_array = luis_detect(rasters, list_filename)
        
        main_ensembles, other_ensembles = luis_ensembles (frame_array, weight_array, rasters)
        
        print len(main_ensembles)
        for k in range(len(main_ensembles)):
            print "...plotting ensemble: ", k
            ax=plt.subplot(3,3,k+1)
            plt.title("Ensemble: "+str(k), fontsize=20)
        
            ax.set_xticks([]); ax.set_yticks([])
            ##Draw all neurons first
            #for i, (y,x,Bmat,thr) in enumerate(zip(y_array,x_array,Bmat_array,thr_array)):
            #    cs = plt.contour(y, x, Bmat, [thr_fixed], colors='black',alpha=0.3)

            #Draw neurons at each modularity 
            unique_indexes = main_ensembles[k]      #Select neurons at this level
            for i in unique_indexes:
                #cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors=colors[k%13],linewidth=15, alpha=1)
                cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors='red',linewidth=15, alpha=1)

            #Draw background neurons 
            unique_indexes = other_ensembles[k]      #Select neurons at this level
            for i in unique_indexes:
                #cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors=colors[k%13],linewidth=15, alpha=1)
                cs = plt.contour(y_array[i], x_array[i], Bmat_array[i], [thr_fixed], colors='black',linewidth=15, alpha=0.5)


        plt.show()
        
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

    




