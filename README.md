#当前问题
之前ResNet50.main等模型文件训练的时候，train里面的miou计算使用的是训练集（训练集测试集没分开），也就是数据泄露，这种场合miou大概是57~58，修复了一下后直接掉到10，并且换了好几个主流模型还是10，我就怀疑是不是dataloader有问题，标签处理之类的。

#dataloader（感觉非常有问题）
old_dataloader → 以前写的dataloader，没有分开训练集和测试集，直接读取的NYUv2的mat文件，没有先转换成RGB/Depth/labels
NYUdataloder → 自己改的，分开了训练集和测试集，不过同样直接读取的NYUv2的mat文件，没有先转换成RGB/Depth/labels
在这个网站https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html下载那个2.8GB的mat文件（已经标注好了的那个），然后替换到路径里就可以了

#模型
ResNet50.main → 以预训练好的ResNet50作为backbone写了个RGB-D双流模型
F_SPConv_encoder4 → 还是以ResNet50为backbone，不过encoder层的最后尝试加了个变种卷积测试效果

#变种卷积
SPConv_4corner → 实验用卷积，可以直接用没问题的
F_SPConv_encoder4 → 

#其他
RGB-D多模态图像语义分割有很多其他的研究论文，感觉他们都是mat提取出rgb/depth/labels/train.txt/test.txt，然后进行一些数据加强操作（比如SA-gate，Asymformer等）
