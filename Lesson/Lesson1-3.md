# **BasicMotion3**

* Keyboard is the most convenient tool for controlling robot in a simulator. We use the `keyboard` package to handle the key event.

```python
import keyboard
```

* Importing time package
  
```python
import time
```

* Importing the `Robot` class for controlling JetBot

```python
from jetbot import Robot
```

* Initializing a class instance of `Robot`

```python
robot = Robot()
```

* We can declare a variable named `speed` to set the base speed of the robot.

```python
speed = 0.3
```

* There is a function named `is_pressed` in keyboard package.We can know which key is pressed and do the correspondingcommand. In the following code, 'w', 'a', 's', and 'd' keysare used to set the motions of forward, left, backward, andright respectively. Pressing the space ' ' can break the infinity loop (while True:). Otherwise, the robot should notmove.

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
