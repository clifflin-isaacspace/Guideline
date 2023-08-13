# **BasicMotion001**

* Importing packages

```python
import time
from jetbot import Robot
```
1. `time` : Control the code execution time
2. `Robot` : Class for controlling JetBot

* Initializing a class instance of `Robot`
  
```python
robot = Robot()
```

* Now that we've created our `Robot` instance we named "robot", we can use this instance to control the robot. To make the robot spin 
counterclockwise at 30% of it's max speed, we can call the following, and the robot can spin counterclockwise.

```python
robot.left(speed=0.3)
```

* To keep running the previous command, we need to use `sleep` function defined in this package. Using `sleep` causes the code execution to block for the specified number of seconds before running the next command. The following method can block the program for half a second.

```python
time.sleep(0.5)
```

* To stop the robot, you can call the `stop` method.  

```
robot.stop()
```

* The basic methods defined in `Robot` class are `left`, `right`, `forward`, and `backward`. Try to plan the trajectory of your own robot.

