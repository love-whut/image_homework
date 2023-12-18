import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/xiaorong/r18-CascadedGroupAttention-MPDIoU/weights/best.pt')
    model.val(data='dataset/NWPU.yaml',
              split='val',
              imgsz=640,
              batch=4,
              device=6,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )