# **基本運動2**

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

* 在之前的範例中，我們看到了如何使用像是`left`、`right`等指令來控制機器人。但如果我們想要分別設定每個馬達的速度呢？好吧，有兩種方法可以做到這一點

* 第一種方法是呼叫 `set_motors` 方法。例如，要在左彎弧上轉動一秒，我們可以設定左馬達為 30%，右馬達為 60%，如下所示。

```python
print('Left arch 1')
robot.set_motors(0.3, 0.6)
time.sleep(1.0)
robot.stop()
```
                                    
* 太棒了！你應該會看到機器人沿著左弧線移動。但實際上，我們還有另一種方法可以完成相同的事情。

* `Robot` 類別具有兩個名為 `left_motor` 和 `right_motor` 的屬性，分別表示每個馬達。這些屬性是 `Motor` 類別的實例，每個實例都包含一個 `value` 屬性。這個 `value` 屬性是一個 `traitlet`，在分配新值時生成事件。在馬達類別中，我們附加了一個函式，該函式在值變更時更新馬達指令。

* 因此，要完成與我們上面所做的完全相同的事情，我們可以執行以下操作。

```python
print('Left arch 2')
robot.left_motor.value = 0.3
robot.right_motor.value = 0.6
time.sleep(1.0)
robot.left_motor.value = 0.0
robot.right_motor.value = 0.0
```
