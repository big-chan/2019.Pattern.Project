# 2019.Pattern.Project

- Colab 용 = 패턴인식_프로젝트.ipynb Step_size =8 에 맞춰져있음
- local 용 - train.py Step_size =4 에 맞춰져있음

## 코드 성능 상황
### DSIFT step size =4


|codebook| VLAD+aug |VLAD+aug+PCA(1024)|
|:---:| :---:|:---:|
|1024| 73% |70%|
|200| 68% |??|

|| SPM | SPM+aug|  
| :---:|:---:|:---:|
|Level 0|42%|47%|
|Level 2|60%|58%|




## Reference

https://github.com/TrungTVo/spatial-pyramid-matching-scene-recognition/blob/master/spatial_pyramid.ipynb

https://github.com/bilaer/Spatial-pyramid-matching-scene-classifier

https://github.com/PrachiP23/Scene-Classification

https://github.com/CyrusChiu/Image-recognition

[VLFEAT](http://www.vlfeat.org/)

https://github.com/jorjasso/VLAD

https://github.com/overshiki/kmeans_pytorch
