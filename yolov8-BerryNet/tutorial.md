# 1. 安装教程
    
    先执行pip uninstall ultralytics 把安装在环境里面的ultralytics库卸载干净
    卸载完成后同样再执行一次,如果出现WARNING: Skipping ultralytics as it is not installed.证明已经卸载干净.
    然后再执行 python setup.py develop 执行这个命令后,对当前ultralytics都会生效.
    具体可看: https://blog.csdn.net/qq_16568205/article/details/110433714

    我的实验环境:
    torch: 1.13.1
    torchvision: 0.14.1

    一些额外的包安装命令:
    pip install timm thop efficientnet_pytorch einops -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -U openmim
    以下主要是使用dyhead必定需要安装的包,如果安装不成功dyhead没办法正常使用!
    mim install mmengine
    mim install "mmcv>=2.0.0"

# 2. 训练 train.py
训练脚本支持导入自己的配置文件基础上导入预训练权重,不需要额外修改,大大方便使用.
weight参数为'',默认就是不载入预训练权重.

    parser.add_argument('--yaml', type=str, default='ultralytics/models/v8/yolov8n.yaml', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='pretrained model path')
    parser.add_argument('--cfg', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--data', type=str, default='ultralytics/datasets/coco128.yaml', help='data yaml path')
    
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--unamp', action='store_true', help='Unuse Automatic Mixed Precision (AMP) training')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    parser.add_argument('--resume', type=str, default='', help='resume training from last checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Adamax', 'NAdam', 'RAdam', 'AdamW', 'RMSProp', 'auto'], default='SGD', help='optimizer (auto -> ultralytics/yolo/engine/trainer.py in build_optimizer funciton.)')
    parser.add_argument('--close_mosaic', type=int, default=0, help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--info', action="store_true", help='model info verbose')
    
    parser.add_argument('--save', type=str2bool, default='True', help='save train checkpoints and predict results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--deterministic', action="store_true", default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds during training for loggers')
    
    # Segmentation
    parser.add_argument('--overlap_mask', type=str2bool, default='True', help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')

    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

# 3. 测试 val.py
验证脚本基本与yolov5使用类似,指定对应参数即可.

    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='training model path')
    parser.add_argument('--data', type=str, default='ultralytics/datasets/coco128.yaml', help='data yaml path')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split to use for validation, i.e. val, test or train')
    parser.add_argument('--project', type=str, default='runs/val', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_json', action="store_true", help='save results to JSON file')
    parser.add_argument('--save_hybrid', action="store_true", help='save hybrid version of labels (labels + additional predictions)')
    parser.add_argument('--conf', type=float, default=0.001, help='object confidence threshold for detection (0.001 in val)')
    parser.add_argument('--iou', type=float, default=0.65, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--half', action="store_true", help='use half precision (FP16)')
    parser.add_argument('--dnn', action="store_true", help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', action="store_true", default=True, help='ave plots during train/val')

# 4. 推理 predict.py
推理脚本支持检测和跟踪一体化,让用户更加方便使用.  
跟踪的时候只需要把mode参数设置为track,tracker参数支持botsort和bytetrack,自行切换配置文件即可.

    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='training model path')
    parser.add_argument('--source', type=str, default='ultralytics/assets', help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'track'], help='predict mode or track mode')
    parser.add_argument('--project', type=str, default='runs/detect', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--show', action="store_true", help='show results if possible')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_conf', action="store_true", help='save results with confidence scores')
    parser.add_argument('--show_labels', action="store_true", default=True, help='show object labels in plots')
    parser.add_argument('--show_conf', action="store_true", default=True, help='show object confidence scores in plots')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=3, help='line width of the bounding boxes')
    parser.add_argument('--visualize', action="store_true", help='visualize model features')
    parser.add_argument('--augment', action="store_true", help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action="store_true", help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--retina_masks', action="store_true", help='use high-resolution segmentation masks')
    parser.add_argument('--boxes', action="store_true", default=True, help='Show boxes in segmentation predictions')
    parser.add_argument('--save', action="store_true", default=True, help='save result')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['botsort.yaml', 'bytetrack.yaml'], help='tracker type, [botsort.yaml, bytetrack.yaml]')

# 5. 模型配置文件
模型配置文件都在ultralytics/models/v8中.
yolov8有五种大小的模型

    YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

怎么指定使用哪一种大小的模型呢？假设我选择的配置文件是yolov8.yaml,我想选择m大小的模型,则train.py中的yaml参数指定为ultralytics/models/v8/yolov8m.yaml即可,同理,如果我想指定s大小的模型,则train.py中的yaml参数指定为ultralytics/models/v8/yolov8s.yaml即可,如果直接设置为ultralytics/models/v8/yolov8.yaml,则默认使用n大小模型.(V5同理)

# 6. 关闭AMP混合精度训练
1. 如果你是使用命令行运行的话,只需要在训练参数中添加--unamp即可.
2. 如果你是直接代码运行的话,找到这个参数parser.add_argument('--unamp', action='store_true', help='Unuse Automatic Mixed Precision (AMP) training'),修改为parser.add_argument('--unamp', action='store_true', default=True, help='Unuse Automatic Mixed Precision (AMP) training')即可.

# 7. 怎么像yolov5那样输出每一层的参数,计算量？
只需要在训练命令指定info参数即可.(v1.5版本中增加输出fuse后的信息)  
需要注意的是指定了info参数不会进行训练,只会输出每一层的一些信息.  

    python train.py --yaml ultralytics/models/v8/yolov8n-fasternet.yaml --info

# 8. 如何替换主干？
可以看项目视频-替换主干示例教程.

# 9. 如何替换yolov5,yolov8中的激活函数？
详细可参考ultralytics/models/v5/yolov5-act.yaml,ultralytics/models/v8/yolov8-act.yaml.  
但是部分激活函数替换后使用info参数输出每一层的参数和计算量的时候有可能会报错,看github上是说thop不支持重用模块,应该是thop的一个bug吧,有解决方案的同学可以联系.(当然这个是不会影响训练)

# 10. 目前自带的一些改进方案(持续更新)

<a id="b"></a>

#### 目前支持的一些block (yolov5默认C3,yolov8默认C2f) 
C2f, C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, C2f_CloAtt, C2f_SCConv, C2f_ScConv, VoVGSCSP, VoVGSCSPC, C3, C3Ghost, C3_CloAtt, C3_SCConv, C3_ScConv

### YOLOV5 (AnchorFree+DFL+TAL) [官方预训练权重github链接](https://github.com/ultralytics/assets/releases)
#### YOLOV5的使用方式跟YOLOV8一样,就是选择配置文件选择v5的即可.
1. ultralytics/models/v5/yolov5-fasternet.yaml

    fasternet替换yolov5主干.
2. ultralytics/models/v5/yolov5-timm.yaml

    使用timm支持的主干网络替换yolov5主干.
3. ultralytics/models/v5/yolov5-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov5中.
4. 增加Adaptive Training Sample Selection匹配策略.

    在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.  
    此ATSS匹配策略目前占用显存比较大,因此使用的时候需要设置更小的batch,后续会进行优化这一功能.
5. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/models/v5/yolov5-AFPN-P345.yaml  
    b. ultralytics/models/v5/yolov5-AFPN-P345-Custom.yaml  
    c. ultralytics/models/v5/yolov5-AFPN-P2345.yaml  
    d. ultralytics/models/v5/yolov5-AFPN-P2345-Custom.yaml  
    其中Custom中的block具体支持[链接](#b)

6. ultralytics/models/v5/yolov5-bifpn.yaml

    添加BIFPN到yolov5中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

7. ultralytics/models/v5/yolov5-C3-CloAtt.yaml

    使用C3-CloAtt替换C3.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C3中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))  

8. ultralytics/models/v5/yolov5-RevCol.yaml

    使用(ICLR2023)Reversible Column Networks对yolov5主干进行重设计.

9. ultralytics/models/v5/yolov5-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov5主干.

10. ultralytics/models/v5/yolov5-C3-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C3融合.

11. ultralytics/models/v5/yolov5-C3-ScConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C3融合.

### YOLOV8
1. ultralytics/models/v8/yolov8-efficientViT.yaml

    (CVPR2023)efficientViT替换yolov8主干.
2. ultralytics/models/v8/yolov8-fasternet.yaml

    fasternet替换yolov8主干.
3. ultralytics/models/v8/yolov8-timm.yaml

    使用timm支持的主干网络替换yolov8主干.
4. ultralytics/models/v8/yolov8-convnextv2.yaml

    使用convnextv2网络替换yolov8主干.
5. ultralytics/models/v8/yolov8-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov8中.
6. ultralytics/models/v8/yolov8-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default)  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
7. ultralytics/models/v8/yolov8-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)
8. ultralytics/models/v8/yolov8-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)
9. ultralytics/models/v8/yolov8-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov8主干.(需要看[常见错误和解决方案的第五点](#a))  
10. ultralytics/models/v8/yolov8-C2f-Faster-EMA.yaml

    使用C2f-Faster-EMA替换C2f.(C2f-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C2f-Faster)
11. ultralytics/models/v8/yolov8-C2f-DBB.yaml

    使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv)
12. 增加Adaptive Training Sample Selection匹配策略.

    在ultralytics/yolo/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.  
    此ATSS匹配策略目前占用显存比较大,因此使用的时候需要设置更小的batch,后续会进行优化这一功能.
13. ultralytics/models/v8/yolov8-slimneck.yaml

    使用VoVGSCSP\VoVGSCSPC和GSConv替换yolov8 neck中的C2f和Conv.
14. ultralytics/models/v8/yolov8-attention.yaml

    可以看项目视频-如何在yaml配置文件中添加注意力层  
    多种注意力机制在yolov8中的使用. [多种注意力机制github地址](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)

15. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/models/v8/yolov8-AFPN-P345.yaml  
    b. ultralytics/models/v8/yolov8-AFPN-P345-Custom.yaml  
    c. ultralytics/models/v8/yolov8-AFPN-P2345.yaml  
    d. ultralytics/models/v8/yolov8-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持这些[结构](#b)

16. ultralytics/models/v8/yolov8-vanillanet.yaml

    vanillanet替换yolov8主干.

17. ultralytics/models/v8/yolov8-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))  

18. ultralytics/models/v8/yolov8-RevCol.yaml

    使用(ICLR2023)Reversible Column Networks对yolov8主干进行重设计.

19. ultralytics/models/v8/yolov8-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov8主干.

20. ultralytics/models/v8/yolov8-C2f-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C2f融合.

21. ultralytics/models/v8/yolov8-C2f-ScConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C2f融合.

# 11. 如何进行断点续训
可以看项目视频-如何进行断点续训.  
train.py中有一个参数是resume,在yolov5中,这个参数只需要设定为True,其就会继续上次没有训练完的任务,但是在yolov8中其是指定一个pt的路径,只需要在resume中指定对应未完成任务中的last.pt即可,如下所示:

    python train.py --weight yolov8n.pt --yaml ultralytics/models/v8/yolov8n.yaml --data /home/hjj/Desktop/dataset/dataset_person/data.yaml --workers 8 --batch 32 --fraction 0.1
    python train.py --data /home/hjj/Desktop/dataset/dataset_person/data.yaml --workers 8 --batch 32 --fraction 0.1 --resume runs/train/exp/weights/last.pt

# 12. 如何计算COCO指标
可以看项目视频-计算COCO指标教程.  

# 13. 绘制曲线对比图
在plot_curve.py中的names指定runs/train中的训练结果名字name即可.  
比如目前runs/train中有exp,exp1,exp2这三个文件夹,plot_curve.py中names中的值为:['exp', 'exp1', 'exp2'],运行后会自动保存为metrice_curve.png和loss_curve.png在当前运行的目录下.

# 常见错误和解决方案(如果是跑自带的一些配置文件报错可以先看看第十大点对应的配置文件是否有提示需要修改内容)
1. RuntimeError: xxxxxxxxxxx does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.....

    解决方案：在ultralytics/yolo/utils/torch_utils.py中init_seeds函数中把torch.use_deterministic_algorithms里面的True改为False

2. ModuleNotFoundError：No module named xxx

    解决方案：缺少对应的包，先把第一大点的安装教程的安装命令进行安装一下，如果还是缺少显示缺少包，安装对应的包即可(xxx就是对应的包).

3. OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.  

    解决方案：https://zhuanlan.zhihu.com/p/599835290

4. 训练过程中loss出现nan.

    可以尝试在训练的时候加上--unamp关闭AMP混合精度训练.

<a id="a"></a>

5. 固定640x640尺寸的解决方案.

    运行train.py中的时候需要在ultralytics/yolo/v8/detect/train.py的DetectionTrainer class中的build_dataset函数中的rect=(True if mode == 'val' else False)中的True改为False.这个是因为EfficientFormerV2固定输入640x640导致的,其他模型可以修改回去.  
    运行val.py不需要改.  
    运行detect.py中的时候需要在ultralytics/yolo/engine/predictor.py找到函数def pre_transform(self, im),在LetterBox中的auto改为False,其他模型可以修改回去.  

# 常见疑问
1. After Fuse指的是什么？

    Fuse是指模型的一些模块进行融合,最常见的就是conv和bn层进行融合,在训练的时候模型是存在conv和bn的,但在推理的过程中,模型在初始化的时候会进行模型fuse,把其中的conv和bn进行融合,通过一些数学转换把bn层融合到conv里面,还有一些例如DBB,RepVGG等等模块支持融合的,这些在fuse阶段都会进行融合,融合后可以一般都可以得到比融合前更快的推理速度,而且基本不影响精度.

2. FPS如何计算？

    在运行val.py后最后会出来Speed: 0.1ms preprocess, 5.4ms inference, 0.0ms loss, 0.4ms postprocess per image这行输出,这行输出就代表了每张图的前处理,推理,loss,后处理的时间,当然在val.py过程中是不需要计算loss的,所以为0,FPS最严谨来说就是1000(1s)/(preprocess+inference+postprocess),没那么严谨的话就是只除以inference的时间,还有一个问题就是batchsize应该设置为多少,其实这行输出就已经是每张图的时间了,但是batchsize还是会对这个时间有所影响,主要是关于并行处理的问题,GPU中可以一次处理多个batch的数据,也可以只处理一个数据,但是处理多batch的数据比处理一个数据的时候整体速度要快,举个例子,比如我有1000张图,我分别设置batchsize为32和batchsize为1,整体运行的时间百分之99都是batchsize为32的快,因此这就导致不同batch输出的时间不同,至于该设置多少来计算FPS,貌似众说纷纭,所以这里我也不好给意见.  
    附上yolov5作者对于FPS和Batch的一个实验链接: https://github.com/ultralytics/yolov5/discussions/6649