## 安装环境

```python
pip uninstall ultralytics
pip install timm thop efficientnet_pytorch einops grad-cam dill -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```


## 训练教程
### 1. 准备训练数据
- 1.1 
  下载NWPU数据集[baiduyun](https://pan.baidu.com/s/1hqwzXeG#list/path=%2F) 
- 1.2
  数据级处理教程[here](https://blog.csdn.net/qq_32575047/article/details/127938321?spm=1001.2014.3001.5506) 

### 2. 开始训练

  ```python
  python train.py --data data/NWPU.yaml --cfg ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --batch-size 4 --epochs 200 --device 0
  ```
## 自带的一些文件说明
1. train.py
    训练模型的脚本
2. main_profile.py
    输出模型和模型每一层的参数,计算量的脚本
3. val.py
    使用训练好的模型计算指标的脚本
4. detect.py
    推理的脚本
5. track.py
    跟踪推理的脚本
6. heatmap.py
    生成热力图的脚本
7. get_FPS.py
    计算模型储存大小、模型推理时间、FPS的脚本
8. get_COCO_metrice.py
    计算COCO指标的脚本
9. plot_result.py
    绘制曲线对比图的脚本  

## 模型配置文件
1.基线模型配置文件

ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml

2.使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention改进rtdetr中的AIFI

ultralytics/cfg/models/rt-detr/rtdetr-CascadedGroupAttention.yaml

## 损失函数修改
ultralytics/models/utils/loss.py

        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True) # GIoU
        # loss[name_giou] = 1.0 - bbox_mpdiou(pred_bboxes, gt_bboxes, xywh=True, mpdiou_hw=2) # MPDIoU






