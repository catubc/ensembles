# Carrillo-Reid et al 2015 SVD based method for computing ensembles
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

def luis_detect(rasters, list_filename):
    
    print rasters.shape
    if os.path.exists(list_filename[:-4]+"_weights.npy"):
        frame_array = np.load(list_filename[:-4]+"_frames.npy")
        weight_array = np.load(list_filename[:-4]+"_weights.npy")
    else:    
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

        #Build cosine similarity matrix
        cos_matrix = np.zeros((len(vec_matrix),len(vec_matrix)), dtype=np.float32)
        for e in range(len(vec_matrix)):
            for f in range(len(vec_matrix)):
                cos_matrix[e,f] = np.dot(vec_matrix[e],vec_matrix[f])/np.linalg.norm(vec_matrix[e])/np.linalg.norm(vec_matrix[f]) # -> cosine of the angle
        
        cos_matrix = (cos_matrix-np.min(cos_matrix))/(np.max(cos_matrix)-np.min(cos_matrix))
        for k in range(len(cos_matrix)):
            cos_matrix[k][np.where(cos_matrix[k]<0.125)[0]]=0
            cos_matrix[k][np.where(cos_matrix[k]>=0.125)[0]]=1
            
        ax=plt.subplot(1,2,1)
        plt.title("Cosine Similarity Matrix", fontsize=20)
        #ax.set_xticks([]); ax.set_yticks([])
        plt.xlabel("High-activity frame #", fontsize=20)
        plt.ylabel("High-activity frame #", fontsize=20)
        plt.imshow(cos_matrix, cmap='Greys')
        plt.ylim(-0.5,len(cos_matrix)-0.5)
        
        
        #Extract SVD matrices
        LA = np.linalg
        a = cos_matrix
        U, s, Vh = LA.svd(a, full_matrices=False)
        n_components = 9
        #assert np.allclose(a, np.dot(U, np.dot(np.diag(s), Vh)))
        ss = s.copy()
        xx = s.copy()
        s[n_components:] = 0
        new_a = np.dot(U, np.dot(np.diag(s), Vh))
        print(new_a)
        thresh = 0.4
        for k in range(len(new_a)):
            new_a[k][np.where(new_a[k]<thresh)[0]]=0
            new_a[k][np.where(new_a[k]>=thresh)[0]]=1
        ax=plt.subplot(1,2,2)
        plt.title("SVD from Similarity Matrix (9 eigenvals)", fontsize=20)
        #ax.set_xticks([]); ax.set_yticks([])
        plt.xlabel("High-activity frame #", fontsize=20)
        plt.ylabel("High-activity frame #", fontsize=20)
        plt.imshow(new_a, cmap='Greys')
        plt.ylim(-0.5,len(cos_matrix)-0.5)
        plt.show()

        #plot first X componenets, e.g. 9 
        thresh = 0.3
        frame_array = []
        weight_array = []
        for k in range(n_components):
            weight_array.append([])
            frame_array.append([])
            ax=plt.subplot(3,3,k+1)
            plt.title("Eigenval: "+str(k), fontsize=20)
            ss=np.zeros(len(s),dtype=np.float32)
            ss[k]=s[k]
            new_a = np.dot(U, np.dot(np.diag(ss), Vh))    
            for p in range(len(new_a)):
                new_a[p][np.where(new_a[p]<thresh)[0]]=0
                new_a[p][np.where(new_a[p]>=thresh)[0]]=1
                
            for r in range(len(new_a)): 
                weight = np.sum(new_a[r])
                if weight>0:
                    weight_array[k].append(weight)
                    frame_array[k].append(r)
                                
            ax.set_xticks([]); ax.set_yticks([])
            plt.imshow(new_a, cmap='Greys')
            plt.ylim(-0.5,len(cos_matrix)-0.5)

        np.save(list_filename[:-4]+"_weights",weight_array)
        np.save(list_filename[:-4]+"_frames",frame_array)

        plt.show()
    
    return frame_array, weight_array

def luis_ensembles(frame_array, weight_array, rasters):
    
    print len(frame_array)
    print len(weight_array)
    
    ensemble_vectors = []
    for k in range(len(frame_array)):
        #print frame_array[k]
        temp = np.zeros(len(rasters),dtype=np.float32)
        for p in range(len(frame_array[k])):
            temp+=rasters[:,frame_array[k][p]]

        ensemble_vectors.append(temp)
        print temp

    main_ensembles = []
    other_ensembles = []
    for e in range(len(ensemble_vectors)):
        indexes = np.where(ensemble_vectors[e]>=(np.max(ensemble_vectors[e])/2.))[0]
        main_ensembles.append(indexes)
        indexes = np.where(np.logical_and(ensemble_vectors[e]>(np.max(ensemble_vectors[e])/4.), ensemble_vectors[e]<(np.max(ensemble_vectors[e])/2.)))[0]
        other_ensembles.append(indexes)

    return main_ensembles, other_ensembles
    

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

    




