# import packages here
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import copy
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np
import torch
import random
import sys
from scipy.sparse import issparse
from sklearn.utils import gen_batches
from sklearn.svm import SVC
import sklearn
import scipy.cluster.vq as vq
import pandas as pd
device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')
import subprocess
use_cuda = torch.cuda.is_available()
from sklearn.decomposition import PCA,IncrementalPCA
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}
dataroot="./input2"
csvpath=os.path.join("Label2Names.csv")
df_data=pd.read_csv(csvpath,header=None)
from sklearn.mixture import GaussianMixture as GMM
dataroottrain=os.path.join("train")
dataroottest=os.path.join("testAll_v2")
##################dataload train
def loadtrain(dataroottrain):
    trainlabel=[]
    traindata=[]


    #import pdb;pdb.set_trace()
    for classname in os.listdir(dataroottrain):
          if classname[0]=='.':
                continue
          classpath=os.path.join(dataroottrain,classname)
          #import pdb;pdb.set_trace()
          if classname=="BACKGROUND_Google":
            labelind=102
          else:
            labelind=(df_data.index[df_data[1]==classname]+1).tolist()[0]  #import pdb;pdb.set_trace()

          for imgname in os.listdir(classpath):
            trainlabel.append(labelind)
            imgpath=os.path.join(classpath,imgname)

            img=cv2.imread(imgpath)
            img=cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA	)
            #descriptor=extract_denseSIFT(img)
            traindata.append(img)
    #traindata=np.array(traindata)
    #trainlabel=np.array(trainlabel)
    return traindata,trainlabel
##################dataload test
def loadtest(dataroottest):
    testlabel=[]
    testdata=[]
    testsort=sorted(os.listdir(dataroottest))
    for classname in testsort:
          if classname[0]=='.':
                continue
          testlabel.append(classname)
          classpath=os.path.join(dataroottest,classname)
          #import pdb;pdb.set_trace()
              
          imgpath=os.path.join(classpath)

          img=cv2.imread(imgpath)
          img=cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
          #descriptor=extract_denseSIFT(img)
          testdata.append(img)
    #testdata=np.array(testdata)
    return testdata,testlabel
#import pdb;pdb.set_trace()
############################################### For Kmeans_GPU
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    #import pdb;pdb.set_trace()
    centers = torch.gather(dataset.to(device_gpu), 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        #import pdb;pdb.set_trace()
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        
        distances = torch.mm(dataset_piece.to(device_gpu), centers_t)
        distances *= -2.0
        #import pdb;pdb.set_trace()
        distances += dataset_norms.to(device_gpu)
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes

# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    #import pdb;pdb.set_trace()
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)
    #import pdb;pdb.set_trace()
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset.to(device_gpu))
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers

def KMeans_GPU(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        #import pdb;pdb.set_trace()
        if torch.equal(codes, new_codes):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes
#########################################################################################################
def extract_sift_descriptors(img):############ For SIFT
    """
    Input BGR numpy array
    Return SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    gray=img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def extract_DenseSift_descriptors(img):###########For DenseSIFT
    """
    Input BGR numpy array
    Return Dense SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    #import pdb;pdb.set_trace()
    gray=img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    # opencv docs DenseFeatureDetector
    # opencv 2.x code
    #dense.setInt('initXyStep',8) # each step is 8 pixel
    #dense.setInt('initImgBound',8)
    #dense.setInt('initFeatureScale',16) # each grid is 16*16
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, gray.shape[0], disft_step_size)
                for x in range(0, gray.shape[1], disft_step_size)]

    keypoints, descriptors = sift.compute(gray, keypoints)

    #xy=[[keypoints[i].pt[j] for j in range(2)] for i in range(len(keypoints))] 
    #xy=np.array(xy).astype("float32")
    #import pdb;pdb.set_trace()
    #descriptors=np.array(descriptors)
    #descriptors=np.concatenate((descriptors,xy),axis=1)
    #import pdb;pdb.set_trace()
    
    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return [keypoints, descriptors]


def build_codebook(X, voc_size):
    """
    Inupt a list of feature descriptors
    voc_size is the "K" in K-means, k is also called vocabulary size
    Return the codebook/dictionary
    """
    features = np.vstack((descriptor for descriptor in X))
    features=torch.from_numpy(features)
    codebook,_ = KMeans_GPU(features,voc_size)
    codebook =codebook.cpu().numpy();
    
    return codebook

from sklearn.decomposition import SparseCoder
def input_vector_encoder(feature, codebook):###########For matching codebook and descriptor
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    #import pdb;pdb.set_trace()
    #coder = SparseCoder(dictionary=codebook, transform_algorithm='threshold')
    #xx = coder.transform(feature)
    
    code, _ = vq.vq(feature, codebook)
    
    #import pdb;pdb.set_trace()
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist

def build_spatial_pyramid_dc(image, descriptor, level):###########for build historgram step 4 level 1
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    s =DSIFT_STEP_SIZE
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
    h = image.shape[0] /s
    w = image.shape[1] / s
    #import pdb;pdb.set_trace()
    if h%1!=0:
        h+=1
    if w%1!=0:
        w+=1
    h=int(h)
    w=int(w)
    
    idx_crop = np.array(range(len(descriptor))).reshape(h,w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 4,4
    #import pdb;pdb.set_trace()  
    shape = (4, 4, 16, 16)
    strides =np.array([4096*2, 64*2, 512, 8])
    
    #import pdb;pdb.set_trace()  
    shape=np.array(shape).astype(int)
    crops = np.lib.stride_tricks.as_strided(idx_crop, shape=shape, strides=strides)
    #import pdb;pdb.set_trace() 
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append([descriptor[idx] for idx in idxs])
    #import pdb;pdb.set_trace()
    return pyramid
def build_spatial_pyramid(image, descriptor, level):###########for build historgram step 4 level 2
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    s =DSIFT_STEP_SIZE
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
    h = image.shape[0] / step_size
    w = image.shape[1] / step_size
    #import pdb;pdb.set_trace()
    if h%1!=0:
        h+=1
    if w%1!=0:
        w+=1
    h=int(h)
    w=int(w)
    
    idx_crop = np.array(range(len(descriptor))).reshape(int(h),int(w))
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw =16, 16
    shape = (2, 2, 32, 32)
    strides = np.array([4096*2*2, 64*2*2, 512, 8])
      
    shape=np.array(shape).astype(int)
    crops = np.lib.stride_tricks.as_strided(idx_crop, shape=shape, strides=strides)
    #import pdb;pdb.set_trace()  
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append([descriptor[idx] for idx in idxs])
    #import pdb;pdb.set_trace()
    return pyramid

def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        
        pyramid +=[descriptor.tolist()]
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        #import pdb;pdb.set_trace()    
        pyramid +=[descriptor.tolist()]
        pyramid +=  build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += [descriptor.tolist()]
        pyramid +=  build_spatial_pyramid(image, descriptor, level=0)
        pyramid +=build_spatial_pyramid_dc(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        #import pdb;pdb.set_trace()
        #code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        #code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 =  np.asarray(code[5:]).flatten()
        return code_level_2#np.concatenate((code_level_0, code_level_1, code_level_2))

def svm_classifier(x_train, y_train, x_test=None, y_test=None,level=1):######## LinearSVM
    best=0
    ##############################################
    '''
    for c in np.arange(0.0001, 0.1, 0.00198):
        clf = LinearSVC(random_state=0, C=c)
        #import pdb;pdb.set_trace()
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)

    predict = clf.predict(x_test)
    return predict
    '''
    ##############################################
    C_range = 10.0 ** np.arange(-3, 3)# np.arange(1, 1000,1)
    #gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(C=C_range.tolist())

    # Grid search for C, gamma, 5-fold CV
    print("Tuning hyper-parameters\n")
    clf = GridSearchCV(LinearSVC(random_state=0), param_grid, cv=5, n_jobs=-2)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    predict=clf.predict(x_test)
    return predict
    
# form histogram with Spatial Pyramid Matching upto level L with codebook kmeans and k codewords
#############################histogram intersection SVM
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp
    return result

def gramSVM(total_train_feature,total_test_feature,train_labels):
    C_range = 10.0 ** np.arange(-6, 6)
    gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(C=C_range.tolist(),gamma=gamma_range.tolist())
    gramMatrix = histogramIntersection(total_train_feature, total_train_feature)
    # scaler.fit(gramMatrix)
    # gramMatrix = scaler.transform(gramMatrix)
    clf = GridSearchCV(svm.SVC(kernel='precomputed'), param_grid, cv=5, n_jobs=-2)
    clf.fit(gramMatrix, train_labels)
    predictMatrix = histogramIntersection(total_test_feature, total_train_feature)
    # predictMatrix = scaler.transform(predictMatrix)
    print(clf.best_params_)
    SVMResults = clf.predict(predictMatrix)
    return SVMResults
#################################################################
#########################################extract VLAD descriptor
def improvedVLAD(X,visualDictionary):

    predictedLabels,_ = vq.vq(X,visualDictionary)
    centers = visualDictionary
    k=1024
   
    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V
###################Kepoint to array
def make_kp_arr(x_kp):
  x_arr=[]
  for keypoints in x_kp:
    i_kp=[]
    for keypoint in keypoints:
      xy=[]
      for i in range(2):
        xy.append(keypoint.pt[i])
      x_arr.append(xy)  
  x_arr=np.array(x_arr).astype("float32")
  return  x_arr
####################SIFT descriptor normalize and geometric extension 
def normalize_extension(x_train_des,x_train_kp):
  img_num=np.array(x_train_des).shape[0]
  des_num=np.array(x_train_des).shape[1]
  x_train_des=np.transpose(x_train_des,(2,0,1))
  x_train_des=x_train_des.reshape(x_train_des.shape[0],x_train_des.shape[1]*x_train_des.shape[2])
  ###########normalize
  sqrt=np.sqrt(np.square(x_train_des).sum(0))
  sqrt[sqrt==0]=1e-5
  nor=1./sqrt
  nor=nor.reshape(1,-1)
  x_train_des=x_train_des*nor

  ############
  ###########PCA
  # encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
  '''
  mean=x_train_des.mean(1).reshape(-1,1)
  new_matrix=x_train_des-mean
  covariance_matrix = np.cov(np.transpose(new_matrix))
  w,v = LA.eig(covariance_matrix)
  U_Truncated = v[:,:2]
  U_Truncated_Transpose = np.transpose(U_Truncated)
  new_dimension_matrix = np.dot(U_Truncated_Transpose,np.transpose(new_matrix))
  newData = np.transpose(new_dimension_matrix)
  val=(-w).argsort()[:2]
  result=v[...,val]
  import pdb;pdb.set_trace();'''
  '''
  
  X = np.dot(x,np.trannspose(x))/ x.shape[1]
  #import pdb;pdb.set_trace()
  import scipy.linalg as la
  D, V = la.eig(X)
  d=D
  d=np.argsort(d)
  d=d[::-1]
  m=100
  perm=np.array(range(127,-1,-1))
  d=d*0.001*np.max(d)
  #import pdb;pdb.set_trace()
  sqrt_=np.sqrt(d[:m])
  sqrt_[sqrt_==0]=1e-5
  V=V[:,perm]
  projection=np.dot(np.diag(1./sqrt_),np.transpose(V[:,:m]))
  '''
  #import pdb;pdb.set_trace()
  #import pdb;pdb.set_trace()
  ###########################################################
  mean=x_train_des.mean(1).reshape(-1,1)
  x_train_des=x_train_des-mean
  ########################################################
  sqrt=np.sqrt(np.square(x_train_des).sum(0))
  sqrt[sqrt==0]=1e-5
  nor=1./sqrt
  nor=nor.reshape(1,-1)
  x_train_des=x_train_des*nor
  #########################################################
  ################geomatric_extension
  x_train_kp=x_train_kp-256/2
  x_train_kp=x_train_kp/256
  x_train_kp=np.transpose(x_train_kp,(1,0))
  x_train_des=np.concatenate((x_train_des,x_train_kp))
  #########################################################
  sqrt=np.sqrt(np.square(x_train_des).sum(0))
  sqrt[sqrt==0]=1e-12
  nor=1./sqrt
  nor=nor.reshape(1,-1)
  x_train_des=x_train_des*nor
  #########################################################
  x_train_des=np.transpose(x_train_des,(1,0))
  x_train_des=x_train_des.reshape(img_num,des_num,130)
  
  return x_train_des
######################################################Main
if not os.path.exists("./traindata.pkl"):
    train_data,train_label=loadtrain(dataroottrain)
    test_data,test_label=loadtest(dataroottest)
    
   
    with open("./traindata.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open("./testdata.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    with open("./trainlabel.pkl", 'wb') as f:
        pickle.dump(train_label, f)
    with open("./testlabel.pkl", 'wb') as f:
        pickle.dump(test_label, f)
else:
    with open("./traindata.pkl", 'rb') as f:
        train_data=pickle.load(f)
    with open("./testdata.pkl", 'rb') as f:
        test_data=pickle.load(f)
    with open("./trainlabel.pkl", 'rb') as f:
        train_label=pickle.load(f)
    with open("./testlabel.pkl", 'rb') as f:
        test_label=pickle.load(f)   

train_data=np.array(train_data)
train_label=np.array(train_label)
train_data=train_data[train_label!=102]#################102 Background 제거
train_label=train_label[train_label!=102]

DSIFT_STEP_SIZE = 4

def PCA_(des):
    pca = PCA(n_components=100, svd_solver='randomized',whiten=True)
    des=pca.fit_transform(des)
    return des

for i in range(0,1):
    print("SPM START%d"%i) 
    if not os.path.exists(os.path.join(dataroot,"/SVMlevel%d.pkl"%i)):
        PYRAMID_LEVEL=i
        name=11## 파일명 변경
        #import pdb;pdb.set_trace()
        if not os.path.exists(os.path.join('spm_lv%d_trainfeature_%d.pkl'%(0,name))):
            
            x_train_feature = [extract_DenseSift_descriptors(img) for img in train_data]
            
            x_test_feature = [extract_DenseSift_descriptors(img) for img in test_data]
        
            x_train_kp, x_train_des = zip(*x_train_feature)
            x_test_kp, x_test_des = zip(*x_test_feature)
            x_train_kp=make_kp_arr(x_train_kp)
            x_test_kp=make_kp_arr(x_test_kp)
          
            print("Done Dense SIFT")
            #import pdb;pdb.set_trace()
            with open(os.path.join('spm_lv%d_trainfeature_%d.pkl'%(0,name)),'wb') as f:
                pickle.dump(x_train_des, f)
            with open(os.path.join('spm_lv%d_testfeature_%d.pkl'%(0,name)),'wb') as f:
                pickle.dump(x_test_des, f)
            with open(os.path.join('spm_lv%d_trainfeature_kp_%d.pkl'%(0,name)),'wb') as f:
                pickle.dump(x_train_kp, f)
            with open(os.path.join('spm_lv%d_testfeature_kp_%d.pkl'%(0,name)),'wb') as f:
                pickle.dump(x_test_kp, f)
        else:
            with open(os.path.join('spm_lv%d_trainfeature_%d.pkl'%(0,name)),'rb') as f:
                x_train_des=pickle.load(f)
                
            with open(os.path.join('spm_lv%d_testfeature_%d.pkl'%(0,name)),'rb') as f:
                x_test_des=pickle.load(f)
            with open(os.path.join('spm_lv%d_trainfeature_kp_%d.pkl'%(0,name)),'rb') as f:
                x_train_kp=pickle.load(f)
                
            with open(os.path.join('spm_lv%d_testfeature_kp_%d.pkl'%(0,name)),'rb') as f:
                x_test_kp=pickle.load(f)
                
        x_train_des=normalize_extension(x_train_des,x_train_kp)
        x_test_des=normalize_extension(x_test_des,x_test_kp)
        #x_train_des=[PCA_(i) for i in x_train_des]
        #x_test_des=[PCA_(i) for i in x_test_des]
        if not os.path.exists(os.path.join('spm_lv%d_codebook_200_%d.pkl'%(0,name))):
            codebook = build_codebook(x_train_des, 1024)
            with open(os.path.join('spm_lv%d_codebook_200_%d.pkl'%(0,name)),'wb') as f:
                pickle.dump(codebook, f)
        else:
            with open(os.path.join('spm_lv%d_codebook_200_%d.pkl'%(0,name)),'rb') as f:
                codebook=pickle.load(f)
            print("Load Codebook")
            
        x_spm_train=[improvedVLAD(i,codebook)for i in x_train_des]  
        x_spm_test=[improvedVLAD(i,codebook) for i in x_test_des] 
        x_spm_train = np.array(x_spm_train)
        x_spm_test = np.array(x_spm_test)
        
        import pdb;pdb.set_trace()
        #x_train_des=[PCA_(i) for i in x_train_des]
        #if not os.path.exists('./SPMfeaturelevel%d_train%d.pkl'%(i,i)):
        '''
        x_spm_train = [spatial_pyramid_matching(train_data[i],
                                                x_train_des[i],
                                                codebook,
                                                level=PYRAMID_LEVEL)
                                                for i in range(len(train_data))]
        x_spm_train = np.asarray(x_spm_train)
        x_spm_test = [spatial_pyramid_matching(test_data[i],
                                               x_test_des[i],
                                               codebook,
                                               level=PYRAMID_LEVEL) for i in range(len(test_data))]
        x_spm_test = np.asarray(x_spm_test)'''
        print("Load SPM")    
        
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        '''
        pca = IncrementalPCA(n_components=1024).fit(x_spm_train)
        
        x_spm_train=pca.transform(x_spm_train)
        x_spm_test=pca.transform(x_spm_test)'''
        print ("Done PCA")
        #SVM_Classify(x_spm_train,train_label,train_label,test_label,i)
        #predict=SVM_Classify(x_spm_train, train_label, x_spm_test, test_label,i)
        #predict =svm_classifier(fv_train, train_label,fv_test )          ############Linear SVM
        predict =svm_classifier(x_spm_train, train_label,x_spm_test )          ############Linear SVM
        
        #predict=gramSVM(fv_train,fv_test,train_label)                      #########intersection
        
        #predict=gramSVM(x_spm_train,x_spm_test,train_label)                      #########intersection
        
        predict=predict.reshape(-1,1)
        
        test_label=np.array(test_label)
        test_label=test_label.reshape(-1,1)
        #import pdb;pdb.set_trace()
        total_result=np.hstack([test_label,predict])
        
        df = pd.DataFrame(total_result,columns=["Id","Category"])
        df.to_csv("./result_dchan_level.csv",index=False,header=True)
        #kaggle competitions submit -c 2019-ml-finalproject -f result_dchan_level.csv -m DCHAN_FINAL
        subprocess.call(["kaggle","competitions","submit","-c", "2019-ml-finalproject", "-f","result_dchan_level.csv","-m", "DCHAN_FINAL"])
