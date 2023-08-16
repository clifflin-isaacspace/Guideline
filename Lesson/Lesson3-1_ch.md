# **道路循跡 - 資料收集**

* 如果您已經運行了碰撞避免範例，您應該熟悉以下三個步驟：

1. 數據收集
2. 訓練
3. 部署

* 在這裡，我們將完全做相同的事情！不過，與其進行分類，您將學習一種不同的基本技術，「回歸」，我們將使用這種技術使 JetBot 能夠跟隨道路（或實際上是任何路徑或目標點）。

1. 在路徑上的不同位置放置 JetBot（偏離中心、不同角度等）。記住從碰撞避免中學到的，數據的變化是關鍵！
2. 顯示機器人的實時攝影機畫面。
3. 使用滑鼠放置一個「綠點」，該點對應於我們希望機器人在圖像上行進的目標方向。
4. 將這個「綠點」的 X、Y 值與機器人攝影機的圖像一起儲存。
       
* 然後，在訓練腳本中，我們將訓練一個神經網絡來預測我們標籤的 X、Y 值。在實時演示中，我們將使用預測的 X、Y 值來計算近似的轉向值（這不是「確切」的角度，因為那需要圖像校準，但它大致與角度成比例，所以我們的控制器能夠正常工作）。

* 那麼，你如何確定在這個範例中準確地放置目標呢？這裡有一個我們認為有幫助的指南：

1. 查看來自攝影機的實時視頻畫面。
2. 想像機器人應該遵循的路徑（嘗試近似需要避免偏離道路等的距離）。
3. 將目標放置在此路徑上，盡可能遠離機器人，這樣機器人可以直接前往目標而不會偏離道路。例如，如果我們在一條非常直的道路上，可以將它放在地平線上。如果我們在一個急轉彎上，它可能需要更靠近機器人，以免越界。

* 假設我們的深度學習模型按預期運行，這些標記指南應確保：

1. 機器人可以安全地朝著目標前進（不會越界等）。
2. 目標將持續沿著我們想像的路徑前進。

* 我們得到的是一個在我們所期望的軌跡上移動的「carrot on a stick」。深度學習決定在哪裡放置這個「carrot」，而 JetBot 則只是跟隨它。

* 現在，我們需要導入 `Robot` 類別和 `Camera` 類別，以控制機器人並從攝影機中獲取影像幀。
                 
```python
from jetbot import Robot, Camera
```

* 要顯示和儲存圖像，我們使用名為 `cv2` 的 OpenCV 套件來執行這些操作。
                                    
```python
import cv2
```

* 由於我們需要建立資料夾來儲存數據，我們可以使用 `os` 套件來執行這個操作。
                                    
```python
import os
```

* 在模擬器中，鍵盤是控制機器人最方便的工具。我們使用 `keyboard` 套件來處理按鍵事件。

                                    
```python
import keyboard
```

                                    
* 為了確保我們不會重複使用任何文件名稱（即使跨不同的機器！），我們將在 Python 中使用 `uuid` 套件，該套件定義了 `uuid1` 方法來生成唯一的識別符。這個唯一的識別符是從類似當前時間和機器地址的信息生成的。
                                    
```python
from uuid import uuid1
```

                                    
* 為了方便起見，讓我們定義一個物件來存儲滑鼠的狀態。
                                    
```python
class Mouse(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.clicked = False

```

* 接下來，我們定義一個回調函數來更新滑鼠的狀態。
                                    
```python
def onMouse(event, x, y, flags, param):
    param.clicked = (event == 1)
    param.x = x
    param.y = y


```

* 同樣地，讓我們創建一個資料夾來儲存數據。
                                    
```python
DATASET_DIR = 'dataset_xy'

try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('Directories not created because they already exist')

```

* 初始化攝影機和機器人。
                                    
```python
camera = Camera.instance(width=224, height=224)
robot = Robot()

```

* 我們可以聲明一個名為 `speed` 的變數來設定機器人的基本速度。

                                    
```python
speed = 0.2
```

* 宣告一個滑鼠物件並設定回調函式。我們必須將回調函式綁定到一個命名窗口，該窗口將用於顯示攝影機的預覽。
                                    
```python
mouse = Mouse()
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse, mouse)

while True:

```

* 我們需要有攝影機畫面的副本，因為我們將在圖像上繪製小工具，並且不希望更改原始畫面。

                                    
```python
    image = camera.value.copy()
```

* `pt1` 是圖像底部中心的點，`pt2` 是滑鼠的位置。
    
```python
    pt1 = (camera.width // 2, camera.height)
    pt2 = (mouse.x, mouse.y)

```

* 繪製 `pt1`、`pt2` 以及它們之間的連線。
                                    
```python
    image = cv2.circle(image, pt1, 8, (0, 0, 255), 4)
    image = cv2.circle(image, pt2, 8, (0, 255, 0), 4)
    image = cv2.line(image, pt1, pt2, (255, 0, 0), 4)
```


* 如果滑鼠被點擊，記錄當前滑鼠位置並保存圖像。滑鼠位置將出現在圖像的名稱中。
                                    
```python
    if (mouse.clicked):
        mouse.clicked = False
        name = 'xy_%03d_%03d_%s.jpg' % (mouse.x, mouse.y, uuid1())
        print('save:', name)
        cv2.imwrite(DATASET_DIR + '/' + name, camera.value)

    cv2.imshow('camera', image)
    cv2.waitKey(1)
```

<p align="center">
  <img alt="VS Code in action" src="https://raw.githubusercontent.com/clifflin-isaacspace/Guideline/main/Lesson/05.bmp" width=480>
</p>

* 跟 `basic_motion_003.py` 一樣，我們使用鍵盤來控制機器人。此外，我們也可以使用鍵盤對圖像進行標記並保存。我們使用 'f' 鍵來標記為 "free"，使用 'b' 鍵來標記為 "blocked"。
                                    
```python
    if   (keyboard.is_pressed(' ')):
        break
    elif (keyboard.is_pressed('w')):
        robot.forward(speed)
    elif (keyboard.is_pressed('s')):
        robot.backward(speed)
    elif (keyboard.is_pressed('a')):
        robot.left(speed)
    elif (keyboard.is_pressed('d')):
        robot.right(speed)
    else:
        robot.stop()

camera.stop()
cv2.destroyAllWindows()
```
