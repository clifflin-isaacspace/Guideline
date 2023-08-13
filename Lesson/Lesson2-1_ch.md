# **視覺避障 1**

* 如果你能夠運行基本的運動程式，希望你會喜歡看到製作 JetBot 移動起來有多麼簡單！這真的很酷！但更酷的是，讓 JetBot 自己移動起來！

* 這是一個非常困難的任務。它有許多不同的方法，但整個問題通常被分解成更簡單的子問題。可以說，其中一個最重要的子問題是解決防止機器人進入危險狀況的問題！我們稱之為「碰撞避免」。

* 在這組程式中，我們將嘗試使用深度學習和一個非常多用途的傳感器來解決這個問題，這個傳感器就是攝影機。你將會看到，透過神經網絡、攝影機和NVIDIA Jetson Nano，我們可以教導機器人一個非常有用的行為！

* 我們採取的避免碰撞方法是在機器人周圍創建一個虛擬的「安全氣泡」。在這個安全氣泡內，機器人可以在不撞到任何物體（或其他危險情況，如掉落懸崖）的情況下旋轉。

* 當然，機器人受限於其視野內的東西，我們無法防止物體被放置在機器人後面等等。但是，我們可以防止機器人進入這些情境。

* 我們將採取的方法非常簡單：

1. 首先，我們將手動將機器人放入其「安全氣泡」被侵犯的情境中，並將這些情境標記為「blocked」。我們會保存機器人所看到的快照，以及這個標籤。

<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/02.bmp" width="480" title="Feature_map" /></p>

2. 其次，我們將手動將機器人放入安全情況下可以稍微向前移動的情境中，並將這些情境標記為「free」。同樣地，我們會保存快照以及這個標籤。

<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/01.bmp" width="480" title="Feature_map" /></p>

* 這就是我們在這個程式中要做的全部內容：資料收集。一旦我們擁有大量的圖像和標籤，我們將使用這些數據來訓練神經網絡，根據機器人所看到的圖像預測其「安全氣泡」是否被侵犯。最終，我們將使用這個預測來實現簡單的碰撞避免行為。

* 匯入`time`套件

```python
import time
```

* 現在，我們需要導入 `Robot` 類別和 `Camera` 類別，以控制機器人並從攝影機中獲取影像幀。
                                    
```python
from jetbot import Robot, Camera
```

* 要顯示和儲存圖像，我們使用名為 `cv2` 的 OpenCV 套件來執行這些操作。

```python
import cv2
```

* 由於我們需要建立儲存資料的資料夾，我們可以使用 `os` 套件來執行這個動作。
                                    
```python
import os
```


* 在模擬器中，鍵盤是控制機器人最方便的工具。我們使用 `keyboard` 套件來處理按鍵事件。

                                    
```python
import keyboard
```

                                  
* 為了確保我們不重複使用任何文件名稱（甚至跨不同的機器！），我們將在 Python 中使用 `uuid` 套件，該套件定義了 `uuid1` 方法以生成唯一的識別符。這個唯一的識別符是從類似當前時間和機器地址的信息生成的。

                                    
```python
from uuid import uuid1
```

                                    
* 為了方便起見，我們可以定義一個函式來處理保存圖像的進程。
                                    
```python
def save_snapshot(directory, image):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    cv2.imwrite(image_path, image)
```

    
* 首先，讓我們創建幾個目錄，用來存儲所有的數據。我們將創建一個名為 `dataset` 的資料夾，該資料夾將包含兩個子資料夾 `free` 和 `blocked`，我們將在其中放置每種情境的圖像。
                                    
```python
blocked_dir = 'dataset/blocked'
free_dir    = 'dataset/free'

try:
    os.makedirs(free_dir)
    os.makedirs(blocked_dir)
except FileExistsError:
    print('Directories not created because they already exist')


```


* 讓我們初始化我們的攝影機。由於我們的神經網絡需要一個 224x224 像素的圖像作為輸入，我們將設定我們的攝影機為該大小，以最小化我們的數據集文件大小（我們已經測試過這在這個任務中是有效的）。在某些情況下，可能更好的方法是以較大的圖像尺寸收集數據，然後在之後縮小到所需的尺寸。
                                    
```python
camera = Camera.instance(width=224, height=224)
```


* 初始化一個 `Robot` 類別的實例。
                                    
```python
robot = Robot()
```

                                    
* 我們可以宣告一個名為 speed 的變數，來設定機器人的基本速度。

                                    
```python
speed = 0.2
                                    
while True:

```

                                    
* 我們可以從攝影機獲取圖像幀，並通過 OpenCV 小工具顯示它。給定窗口的名稱和一個圖像，`cv2.imshow` 函式可以在該窗口中顯示圖像。在以下的程式碼中，圖像將會在一個名為 'camera' 的窗口中繪製。`cv2.waitKey` 函式會阻塞程式碼一段時間，等待按鍵事件並且同時交換緩衝區以顯示圖像。例如，`cv2.waitKey(1)` 會阻塞程式碼 1 毫秒並在窗口中顯示圖像。

                                    
```python
    image = camera.value
    cv2.imshow('camera', image)
    cv2.waitKey(1)

```

                                    
* 與 `basic_motion_003.py` 類似，我們使用鍵盤來控制機器人。此外，我們還可以使用鍵盤來為圖像標記並保存它。我們使用 'f' 鍵來標記為「free」，使用 'b' 鍵來標記為「blocked」。

                                    
```python
    if   (keyboard.is_pressed(' ')):
        break
    elif (keyboard.is_pressed('f')):
        save_snapshot(free_dir, image)
        print('free_count', len(os.listdir(free_dir)))
        time.sleep(0.1)
    elif (keyboard.is_pressed('b')):
        save_snapshot(blocked_dir, image)
        print('blocked_count', len(os.listdir(blocked_dir)))
        time.sleep(0.1)
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
