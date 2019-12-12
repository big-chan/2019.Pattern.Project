# 2019.Pattern.Project

- Colab 용 = 패턴인식_프로젝트.ipynb Step_size =8 에 맞춰져있음
- local 용 - train.py Step_size =4 에 맞춰져있음

train.py 안에도 BoW 와 VLAD 를 다 사용할 수 있도록 주석 처리를 해두었습니다만 일단은 VLAD에 맞춰서 되어있습니다. 
train_BOW.py 를 실행 시키면 BoW Level 0 부터 Level 2 까지 다 실행 되도록 했습니다. 

### Aug 란 

- 제 코드 내에서 Normalize 와 Geomatric Extension 을 합쳐서 적용하는 것입니다. 이것의 적용을 통해서 배아수BoW 와VLAD 에서 성능향상을 확인했습니다.

## 코드 성능 상황

- codebooksize=200

|      | Step_size | +aug| +PCA| 성능 |  
| :----:|:---------:|:---:|:---:|:---:|
|Level 0|    4    |X   |  X | 39% |
|Level 0| 4 |O    | X | 47% |
|Level 0|    8    |X   |  X | 42% |
|Level 0|8        |O    | X | 48% |
|Level 0|    4    |X   |  O(PCA_SIFT) | 10이하 |
|Level 0|4        |O    | O(PCA_SIFT) | 10이하 |
|Level 0|    8    |X   |  O(PCA_SIFT) | 10이하 |
|Level 0|8        |O    | O(PCA_SIFT) | 10이하 |
|Level 2|    4    |X   |  X | 60% |
|Level 2|4        |O    | X | 57% |
|Level 2|    8    |X   |  X | 58% |
|Level 2|8        |O    | X | 54% |

- Level 0 에서 aug 을 진행 했을 때 step_size 에 상관 없이 성능이 향상 되는 것을 확인했다. 하지만 SPM 을 진행 하는 순간 성능이 aug 했을 때 더욱 떨어지는 것을 확인할 수 있다. 이를 통해서 SPM 을 쌓는거나 descriptor 를 자르는 과정에서 원복을 완료 하지 못한 부분이 있다고 추측할 수 있다. 하지만 Level 0 에 대해서는 논문을 원복을 확인한 후 aug의 성능향상을 확인 했으니 level 0 에서 진행 하는 VLAD에 aug 적용시 성능을 향상 시킬 수 있다고 생각 가능해진다.

### DSIFT step size =4


|codebook| VLAD | VLAD+aug |VLAD+aug+PCA(1024)|
|:---:|:---:| :---:|:---:|
|200|60% | 68% |??|
|600 |63%(stepsize 8)| ???| ??|
|1024|??| 73% |70%|

- 실험을 아직 다 못한 부분 들이 있어 표를 채우진 못했습니다만 지표를 볼떄 확실히 BOW 에서 의 경향성 대로 베이스 만 잘 잡혀 있다면 aug 를 적용했을 떄 성능이 향상됨을 확인 할 수 있습니다.

## SPM 원복 실패(예측)

```
def build_spatial_pyramid_dc(image, descriptor, level):
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
def build_spatial_pyramid(image, descriptor, level):
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
```
- 이는 SPM 을 위해 Descriptor를 위에서 부터 1/16과 1/4 하는 코드이다. 파라미터 값을 이해를 제대로 하지 못한 상태로 적용해서 그런지 원복이 원인인것같다.


## Reference

https://github.com/TrungTVo/spatial-pyramid-matching-scene-recognition/blob/master/spatial_pyramid.ipynb

https://github.com/bilaer/Spatial-pyramid-matching-scene-classifier

https://github.com/PrachiP23/Scene-Classification

https://github.com/CyrusChiu/Image-recognition

[VLFEAT](http://www.vlfeat.org/)

https://github.com/jorjasso/VLAD

https://github.com/overshiki/kmeans_pytorch
