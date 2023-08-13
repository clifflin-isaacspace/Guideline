# **基本運動1**

* 匯入套件
  
```python
import time
from jetbot import Robot
```

1. `time` : 控制程式執行時間
2. `Robot` : 用於控制JetBot的套件

* 初始化一個 Robot 類別的實例。
  
```python
robot = Robot()
```

* 現在我們已經建立了我們命名為 "robot" 的 Robot 實例，我們可以使用這個實例來控制機器人。要使機器人以其最大速度的 30％ 逆時針旋轉，我們可以呼叫以下方法，機器人就可以逆時針旋轉。

```python
robot.left(speed=0.3)
```

* 為了繼續執行前面的命令，我們需要使用這個套件中定義的 sleep 函式。使用 sleep 會導致程式碼執行在執行下一個命令之前阻塞指定的秒數。下面的方法可以將程式封鎖半秒鐘。

```python
time.sleep(0.5)
```

* 要停止機器人，您可以呼叫 stop 方法。
  
```
robot.stop()
```

* Robot 套件中定義的基本方法包括 left、right、forward 和 backward。試著規劃您自己機器人的軌跡。 

