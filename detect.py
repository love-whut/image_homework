import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/data/war/model/RTDETR-main/runs/train/r18_200_baseline-new/weights/best.pt') # select your model.pt path
    model.predict(source='/data/war/dataset/NWPU/NWPU_YOLO/images/val',
                  project='runs/detect',
                  name='exp',
                  save=True,
                  show_conf=False,
                  # visualize=True # visualize model features maps
                  )