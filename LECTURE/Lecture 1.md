# 9.17 **Lecture 1**

## TOPICS：
 - Solving problems using computation
 - Python programming language
 - Organizing modular programs
 - Some simple but important algorithms
 - Algorithmic complexity
 ### references：
 - Read tutorial & reference at https://docs.python.org/
--- 
 - Chinese: https://www.runoob.com/python3/python3-tutorial.html
 - Basics: https://github.com/jackfrued/Python-100-Days
 - Book: Eric Matthes, Python Crash Course, 3rd ed., 2023
 ---


## ALGORITHM 算法
1. Sequence of simple steps 
2. Flow of control process that specifies when each step is executed 
3. A means of determining when to stop 
>"COMPUTERS are MACHINES that EXECUTE ALGORITHMS."
---

## PYTHON BASICS
- A program is a sequence of definitions and commands
     - Definitions evaluated
     - Commands executed by Python interpreter in a shell
- Can be typed directly in a shell or stored in a file that is ready into the shell and evaluated


### OBJECTS 对象: 
- Programs manipulate data objects
- Objects have a type that defines the kinds of things programs can do to them
    - Scalar 标量 (cannot be subdivided)：数值，布尔值
    - Non-scalar (have internal structure that can be accessed)：列表，字典，字符串
---
#### SCALAR OBJECTS：
- int – represent integers, ex. 5, -100
- float – represent real numbers, ex. 3.27, 2.0
- bool – represent Boolean values True and False
- NoneType – special and has one value, None
- Can use type() to see the type of an object：
    ##### TYPE CONVERSIONS (CASTING) 类型转换
 - ```float(3)``` casts the int 3 to float 3.0 `float(3)`将整数3转换为浮点数3.0
 - ```round(3.9)``` returns the int 4 `round(3.9)`返回整数4
---
### EXPRESSIONS 表达式
Combine objects and operators to form expressions 将对象和操作符组合成表达式   
An expression has a value, which has a type 表达式有值且有类型
 - 3+2 has value 5 and type int 3+2的值为5，类型为int
 - 5/3 has value 1.666667 and type float 5/3的值为1.666667，类型为float
---
### *OPERATORS on int and float 整数与浮点数的运算符
 - `i+j` → the sum i+j 求和
 - `i-j` → the difference i-j 求差
 - `i*j` → the product i*j 求积
 - `i/j` → division i/j 除法
 - `i//j` → floor division i//j 取整除
 - `i%j` → the remainder when i is divided by j i%j 求余数
 - `i**j` → i to the power of j i**j 求幂
---
### VARIABLES 变量
- Is boundto onesinglevalue at a given time
- Can be bound to an expression (but expressions evaluate to one value!)
    - 使用`=`赋值，把一个value赋给一个variable
#### CHANGE BINDINGS
- Can re-bind variable names using new assignment statements
-  Value for area does not change until you tell the computer to do the calculation again  
```python
 pi = 3.14
 radius = 2.2
 area = pi*(radius**2) 
 radius = radius+1
```
- Swap values of x and y:
```python
x = 1
y = 2
t = x
x = y
y = t
```
or
```python
x = 1
y = 2
x,y = y,x
```
---
### STRINGS
- Think of a str as a sequence of case sensitive characters  
Enclosed in quotation marks or single quotes
```python
a = "me"  
z = 'you'
```
- Concatenate and repeat strings:
```python
b="myself" 
c=a+b
d=a+""+b 
silly=a*3
```
#### STRING OPERATIONS
- `len()` is a function used to retrieve the length of a string in the parentheses  
    python字符串默认结尾不含`\0`，但输入`\0`可被计入长度
```python
s = "abc"
len(s)  #输出3
chars = len(s)
```
#### SLICING to get ONE CHARACTER IN A STRING 切片
- Square brackets used to perform indexing into a string to get the value at a certain index/position  
`s = "abc"`  
`s[0]` -> a  
`s[-1]` -> c
- SLICING to get a SUBSTRING 子字符串
    - Can slice strings using **[start : stop : step]**（default step = 1,且为正)
    - Get characters at start
    - up to and including stop-1(左闭右开)
    - taking every step characters
    `s = "abcdefgh"`  
    `s[3:6]`   → evaluates to `"def"`,same as `s[3:6:1]`  
    `s[3:6:2]` → evaluates to `"df"`  
    `s[:]`     → evaluates to `"abcdefgh"`, same as `s[0:len(s):1] `  
    `s[::-1]`  → evaluates to `"hgfedcba"`,逆序切片，取 索引 start 到 end+1 的字符（左闭右开，逆序）    
    `s[4:1:-2]`→ evaluates to `"ec"`  

  **TEST:**
`s = "ABC d3f ghi"` -> 空格也占用index，即`s[3] = ' '`
则：  
`s[3:len(s)-1]` -> `' d3f gh'`  
`s[4:0:-1]` -> `'d CB'`  
`s[6:3]` -> `''`(若 start >= end 且 step 为默认正序（step=1），则无有效字符，返回空字符串。)  


#### IMMUTABLE STRINGS
Strings are “immutable” – cannot be modified  
You can create new objects that are versions of the original one  
Variable name can only be bound to one object  
```python
s = "car"
s[0] = 'b'
s = 'b'+s[1:len(s)]
```
---

## **TAKEAWAY MESSAGES**
- Objects
    - Objects in memory have types.
    - Types tell Python what operations you can do with the objects.
    - Expressions evaluate to one value and involve objects and operations
    - Variables bind names to objects.
    - `=` sign is an assignment, for ex. var = type(5*4)
- Programs
    - Programs only do what you tell them to do.
    - Lines of code are executed in order.
    - Good variable names and comments help you read code later.
- String & Slice
    - `len()`to get the length of the string
    - `[start : end : step]` to get a slice of the string  
      *the end index itself would not be printed*  
      *if the end index is larger than the start index and the step is not negative,output would be `''`*  
---