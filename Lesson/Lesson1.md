-[What Is Python](#WhatIsPython)

-[Basic Motion 001](#BasicMotion001)


# **WhatIsPython**

Python is a high-level, interpreted, and general-purpose programming language that was created by Guido van Rossum and first released in 1991. It is known for its simplicity, readability, and versatility, making it one of the most popular programming languages in the world. Python's design philosophy emphasizes code readability, making it easier for programmers to express their ideas clearly and concisely.

Key features of Python include:

1. Easy-to-learn syntax: Python uses indentation to define code blocks, making it more readable and forcing developers to write clean and organized code.

2. Interpreted language: Python code is executed line by line by an interpreter, which means you don't need to compile the code explicitly before running it.

3. Cross-platform: Python is available and runs on various operating systems, such as Windows, macOS, Linux, and others.

4. High-level data structures: Python provides built-in data structures like lists, dictionaries, and tuples, making it effective for handling complex data.

5. Large standard library: Python comes with a rich set of modules and libraries that simplify common programming tasks, ranging from working with strings and files to web development and networking.

6. Object-oriented programming support: Python supports object-oriented programming principles like encapsulation, inheritance, and polymorphism.

7. Dynamic typing: Python uses dynamic typing, meaning you don't need to declare variable types explicitly; their types are determined at runtime.

8. Extensibility: Python can be easily extended with C/C++ code, allowing developers to optimize performance-critical sections of the program.

Python is widely used in various domains, including web development, data analysis, artificial intelligence, scientific computing, automation, scripting, and more. Its popularity can be attributed to its large community, extensive libraries, and widespread adoption in various industries.

# **BasicMotion001**

* Importing packages
<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/1-1-1.bmp" width="320" title="1-1-1" /></p>

1. `time` : Control the code execution time
2. `Robot` : Class for controlling JetBot

* Initializing a class instance of `Robot`
<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/1-1-2.bmp" width="160" title="1-1-2" /></p>

* Now that we've created our `Robot` instance we named "robot", we can use this instance to control the robot. To make the robot spin 
counterclockwise at 30% of it's max speed, we can call the following, and the robot can spin counterclockwise.

<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/1-1-3.bmp" width="180" title="1-1-3" /></p>

* To keep running the previous command, we need to use `sleep` function defined in this package. Using `sleep` causes the code execution to block for the specified number of seconds before running the next command. The following method can block the program for half a second.

<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/1-1-4.bmp" width="180" title="1-1-4" /></p>

* To stop the robot, you can call the `stop` method.
  
<p float="left"><img src="https://github.com/clifflin-isaacspace/Guideline/blob/main/Lesson/1-1-5.bmp" width="180" title="1-1-5" /></p>

* The basic methods defined in `Robot` class are `left`, `right`, `forward`, and `backward`. Try to plan the trajectory of your own robot.

