# **道路循跡 2**

* 首先，我們導入所有需要的套件。
                                    
```python
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import time

from jetbot import Robot, Camera
```

## 載入已訓練的模型
*** 

* 我們將假設您已經有了 `best_steering_model_xy.pth`。現在，您應該初始化 PyTorch 模型並從 `best_steering_model_xy.pth` 中載入已訓練的權重。
                                    
```python
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('best_steering_model_xy.pth'))

```

* 同樣地，我們使用 CPU 來運行模型。
                                    
```python
device = torch.device('cpu')
model = model.to(device).eval()

```


## 建立預處理函數
***							
								
* 我們訓練模型的格式與攝影機的格式不完全相符。為了解決這個問題，我們需要進行一些預處理。這包括以下步驟：

1. 從 BGR 轉換為 RGB。
2. 從 HWC 佈局轉換為 CHW 佈局。
3. 使用與訓練期間相同的參數進行正規化。我們的攝影機提供的是 [0,255] 範圍的圖像，而訓練時載入的圖像則在 [0,1] 範圍內，所以我們需要乘以 255.0 進行縮放。
4. 將數據從 CPU 記憶體轉移到您選擇的設備。
5. 添加批次維度。

                       
```python
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x
```

## Jetbot與模型部署
*** 

* 此外，我們需要初始化機器人及其攝影機。
                                    
```python
robot = Robot()
camera = Camera.instance(width=224, height=224)

```

                                    
* 現在，我們將定義一些參數來控制 JetBot：

1. 速度控制（speed）：設定一個值來啟動您的 JetBot。在這裡，我們設定了一個默認值為 20%。

2. 轉向增益控制（steering_gain 和 steering_dgain）：如果您看到 JetBot 在運行時搖擺不定，您需要減小 "steering_gain" 或 "steering_dgain" 直到它平穩。

3. 轉向偏差控制（steering_bias）：如果您看到 JetBot 偏向於賽道的極端右側或極端左側，您應該調整此變數，直到 JetBot 開始在中心處跟隨線條或賽道。這包括馬達偏差以及攝影機偏移。
                                    
```python
speed = 0.2
steering_gain  = 0.09
steering_dgain = 0.1
steering_bias = 0.0

```

* 接下來，我們可以創建一個無限循環，用於執行機器人的例行操作。這個例行操作將執行以下步驟：

1. 預處理攝影機圖像。
2. 執行神經網絡。
3. 計算近似的轉向值。
4. 使用比例/微分控制（PD 控制）來控制馬達。

* 為了可視化轉向角度，我們可以定義一個函數將它繪製在圖像上。
                           
```python
def draw_steering(image, angle, color=(0, 255, 0), radius=30):

    center = (image.shape[1]//2, image.shape[0] - 1)
    theta = angle - 3.14 / 2.0

    rcos = radius * np.cos(theta)
    rsin = radius * np.sin(theta)

    pt1 = (int(0.6 * rcos + center[0]), int(0.6 * rsin + center[1]))
    pt2 = (int(1.4 * rcos + center[0]), int(1.4 * rsin + center[1]))

    cv2.circle(image, center, radius, color, 5)
    cv2.line(image, pt1, pt2, color, 5)

angle_last = 0.0

while True:

```

    
* 我們從攝影機中複製圖像，因為我們希望在圖像上繪製一些信息，並且不希望更改原始圖像。

                                    
```python
    image = camera.value.copy()

    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    angle = np.arctan2(x, y)
    pid = angle * steering_gain + (angle - angle_last) * steering_dgain
    angle_last = angle

    steering = pid + steering_bias

    robot.left_motor.value = max(min(speed + steering, 1.0), 0.0)
    robot.right_motor.value = max(min(speed - steering, 1.0), 0.0)

    draw_steering(image, angle)

    cv2.imshow('camera', image)
    key = cv2.waitKey(1)
    
```
<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/06.bmp" width="480" title="Feature_map" /></p>

* 我們可以按下空格（0x20）鍵來退出這個循環。
                                    
```python
    if (key == 0x20):
        break

camera.stop()
cv2.destroyAllWindows()
```
