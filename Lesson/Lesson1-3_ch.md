# **基本運動3**

* 在模擬器中，鍵盤是最方便的控制工具。我們使用 `keyboard` 套件來處理按鍵事件。
  
```python
import keyboard
```

* 匯入`time`套件
  
```python
import time
```

* 匯入`Robot`套件

```python
from jetbot import Robot
```

* 初始化一個 `Robot` 類別的實例。

```python
robot = Robot()
```

* 我們可以宣告一個名為 `speed` 的變數，來設定機器人的基本速度。

```python
speed = 0.3
```

* 在鍵盤套件中有一個名為 `is_pressed` 的函式。我們可以知道哪個按鍵被按下，並執行相應的指令。在以下的程式碼中，'w'、'a'、's' 和 'd' 鍵分別用於設定向前、向左、向後和向右的動作。按下空白鍵 ' ' 可以中斷無限迴圈 (while True:)。否則，機器人將不會移動。

```python
while True:

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
```
