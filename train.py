import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/extra_models/rtdetr-CascadedGroupAttention-Attention(full).yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/NWPU.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=4,
                device='3',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )