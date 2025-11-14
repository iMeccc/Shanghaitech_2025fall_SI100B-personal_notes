# 9.19 **Lecture 2**
## 课堂live：
### INPUT/OUTPUT:
1. Print 打印输出  
    print()可一次输出多个对象  
    Separate objects using commas to output them separated by spaces  
    Concatenate strings together using + to print as single object 用`+`连接多个字符串
    ```python
    a = "the" 
    b = 3
    c = "musketeers" 
    print(a, b, c) 
    print(a + str(b) + c)
    ```
    ---
2. Input 输入
    `x = input(s)`会先输出字符串s，再读入enter后的内容，将其赋值给x  
    input()返回一个字符串,若想进行数学计算，应用类型转换int(),float()  
    ```python
    text = input("Type anything: ")
    print(5*text)
    ```
    如果希望有空格，则  
    ```python
    print((text+' ')*5)
    ```
    ---
3. F-Stings 格式化
    Character `f` followed by a formatted string literal  
    Expressions bracketed by curly braces { }  
    Expressions in curly braces evaluated at runtime, automatically converted to strings, and concatenated to the string preceding them  
    在字符串前加 f 或 F，然后在 {} 中直接嵌入变量、表达式或函数调用，运行时会自动计算并替换为对应的值。  
    ```python
    num = 3000 
    fraction = 1/3
    print(num*fraction, 'is', fraction*100, '% of', num) 
    print(num*fraction, 'is', str(fraction*100) + '% of', num) 
    print(f'{num*fraction} is {fraction*100}% of {num}')
    ```
    ```python
    name = "Alice"
    age = 25

    # 嵌入变量
    print(f"姓名：{name}，年龄：{age}")  # 输出：姓名：Alice，年龄：25

    # 变量可以是任何类型（数字、列表、对象等）
    scores = [90, 85, 95]
    print(f"成绩列表：{scores}")  # 输出：成绩列表：[90, 85, 95]
    ```
    可嵌入函数调用
    ```python
    def get_full_name(first, last):
    return f"{first} {last}"

    first_name = "John"
    last_name = "Doe"

    # 调用函数并嵌入结果
    print(f"全名：{get_full_name(first_name, last_name)}")  # 输出：全名：John Doe

    # 字符串方法调用
    text = "hello"
    print(f"大写：{text.upper()}")  # 输出：大写：HELLO
    ```
    说明符|作用|示例|输出
    ---|---|---|---
    .nf|保留 n 位小数（四舍五入）|f"π ≈ {3.1415926:.2f}"|π ≈ 3.14
    ,|千位分隔符（用于大数字）|f"人口：{1400000000:,}"|人口：1,400,000,000
    %|转换为百分比（自动乘 100）|f"增长率：{0.052:%}"|增长率：5.200000%
    %n|百分比保留 n 位小数|f"增长率：{0.052:.1%}"|增长率：5.2%

    说明符|作用|示例|输出
    ---|---|---|---
    <n|左对齐，总宽度为 n|f"左对齐：{name:<10}"|左对齐：Alice（假设 name 是 Alice，补足 10 位）
    \>n|右对齐，总宽度为 n|f"右对齐：{name:>10}"|右对齐： Alice	
    ^n|居中对齐，总宽度为 n|f"居中对齐：{name:^10}"|居中对齐： Alice	
    =n|数字符号左对齐，数字右对齐|f"数值对齐：{ -123 :=8}"|数值对齐：- 123	（总宽度 8，负号左靠）
---
### CONDITIONS for BRANCHING
1. COMPARISON OPERATORS
2. LOGICAL OPERATORS on bool 布尔值运算  
    ```python
    not a   →True if a is False 
             False if a is True
    a and b →True if both are True
    a or b  →True if either or both are True
    ```
    e.g.
    ```python
    x = 1  
    y = int(input("Enter a number: "))
    print("does x equal y?", x == y)
    ```
    ---
3. BRANCHING
    >"Indentation matters in Python!"
    ```python
    pset_time = int(input("p time:"))
    sleep_time= int(input("sleep time:"))
    if(pset_time+sleep_time) >24:
    print("impossible!")
    elif(pset_time+sleep_time) >=24:
    print("fullschedule!")
    else:
    leftover=abs(24-pset_time-sleep_time)
    print(leftover,"h of free time!") 
    print("end of day")
    ```
    存在分支，循环，嵌套
    ---
4. CONTROL FLOW:while LOOPS
    ```while 1:```持续执行
    ---
5. RANGE
    generates a sequence of ints,following a pattern  
    `range(start,stop,step)`
    - start:first int generated  
    - stop:controls last int generated (go up to but not including this int)
    - step:used to generate next int in sequence

    `for i in range()`会创建一个list进行遍历
    ```python
    sum = 0
    for x in range(101):
    sum = sum + x
    print(sum)
    ```



