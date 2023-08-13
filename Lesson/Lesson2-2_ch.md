# **Collision Avoidance 2**

* 在這個程式中，我們將使用我們訓練的模型來檢測機器人是否是「free」或「blocked」，以實現機器人的碰撞避免行為。

* 首先，我們導入我們需要的所有套件。
                                    
```python
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import time

from jetbot import Robot, Camera

```

                                    
* 我們假設你已經有了 `best_model.pth`。現在，你應該初始化 PyTorch 模型並從 `best_model.pth` 加載訓練好的權重。

                                    
```python
model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

model.load_state_dict(torch.load('best_model.pth'))

```
                       
* Similarly, we use CPU to run the model.
                                    
```python
device = torch.device('cpu')
model = model.to(device).eval()

```

* 我們訓練模型的格式與攝影機的格式不完全相符。為了解決這個問題，我們需要進行一些預處理。這涉及到以下步驟：

1. 從 BGR 轉換為 RGB。
2. 從 HWC 佈局轉換為 CHW 佈局。
3. 使用與訓練期間相同的參數進行標準化。我們的攝影機提供的是 [0, 255] 範圍的值，而訓練加載的圖像在 [0, 1] 範圍內，所以我們需要乘以 255.0 來進行縮放。
4. 將數據從 CPU 記憶體轉移到您選擇的設備上。
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

* 同時，我們需要初始化機器人及其攝影機，並將基本速度設定為 20%。
                                    
```python
robot = Robot()
camera = Camera.instance(width=224, height=224)
speed = 0.2
```

* 現在，我們可以創建一個無限迴圈，以執行機器人的例行工作。機器人的碰撞避免例行工作應該執行以下步驟：
  
1. 預處理攝影機圖像。
2. 執行神經網絡。
3. 當神經網絡的輸出指示我們被阻塞時，我們會向左轉，否則我們向前移動。

                                    
```python
while True:
```

* 我們從攝影機複製圖像，因為我們希望在圖像上繪製一些信息，而不想更改原始圖像。
                
```python
    image = camera.value.copy()
    x = preprocess(image)
    y = model(x)

```


* 我們應用 `softmax` 函數來對輸出向量進行正規化，使其總和為1（這使其成為概率分布）。
    
                                    
```python
    y = F.softmax(y, dim=1)

    prob_blocked = float(y.flatten()[0])

    if (prob_blocked < 0.5):
        cv2.putText(image, 'Free', (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        robot.forward(speed)
    else:
        cv2.putText(image, 'Blocked', (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        robot.left(speed)

    cv2.imshow('camera', image)
    key = cv2.waitKey(1)


```

<p float="left">
<img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/03.bmp" width="450" title="Feature_map" />
<img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/04.bmp" width="520" title="Feature_map" />
</p>

* 我們可以按下空格鍵（0x20）來退出迴圈。
                                    
```python
    if (key == 0x20):
        break

camera.stop()
cv2.destroyAllWindows()

```
