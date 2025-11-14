# 计算机算法核心英文词汇表 (Alogrithm Problem Core Vocabulary)

本词汇表整合了编程作业中的常见词汇与 AWL 学术词汇表中与计算机科学最相关的部分，旨在帮助你快速理解英文算法题的题意和要求。

---

### **一、 核心概念与数据结构 (Core Concepts & Data Structures)**

| 英文单词 | 中文释义 | 常见语境与说明 |
| :--- | :--- | :--- |
| **algorithm** | **算法** | 程序的核心逻辑和步骤。 |
| **data** | **数据** | 程序处理的信息。 |
| **structure** | **结构** | "Data structure" = 数据结构，指数据的组织方式。 |
| **variable** | **变量** | 用于存储值的名称。 |
| **constant** | **常量** | 值不会改变的量。 |
| **parameter / argument** | **参数 / 实参** | 传递给函数的值。 |
| **integer (int)** | **整数** | 如 `10`, `-5`, `0`。 |
| **float / real** | **浮点数 / 实参** | 如 `3.14`, `-0.5`。题目中 `real` 和 `float` 通常可互换。 |
| **string (str)** | **字符串** | 由字符组成的序列，如 `"hello"`。 |
| **boolean (bool)** | **布尔值** | 只有 `True` 或 `False` 两个值。 |
| **list / array** | **列表 / 数组** | 有序的元素集合，如 `[1, 2, 3]`。 |
| **sublist / subarray** | **子列表 / 子数组** | 列表的一部分，如 `[1, 2, 3]` 中的 `[2, 3]`。 |
| **nested list** | **嵌套列表** | 列表的元素也是列表，如 `[1, [2, 3]]`。 |
| **dictionary (dict) / map** | **字典 / 映射** | 键-值对的集合，如 `{'name': 'Alice'}`。 |
| **grid / matrix** | **网格 / 矩阵** | 二维数据结构，通常用嵌套列表实现。 |
| **element / item** | **元素 / 项目** | 列表、集合等数据结构中的单个成员。 |
| **index / position** | **索引 / 位置** | 元素在列表或字符串中的位置，通常从0开始。 |
| **expression** | **表达式** | 能被计算出一个值的代码，如 `2 + 3`。 |
| **statement** | **语句** | 一个完整的执行指令，如 `x = 5`。 |
| **function / method** | **函数 / 方法** | 需要你编写的代码块。 |

---
### **二、 任务与操作 (Tasks & Operations)**

| 英文单词 | 中文释义 | 常见语境与说明 |
| :--- | :--- | :--- |
| **given...** | **给定...** | 题目描述的开头，告诉你已知的输入信息。 |
| **implement** | **实现** | "Implement a function" = 实现一个函数。将逻辑写成代码。 |
| **compute / calculate** | **计算** | "Compute the sum" = 计算总和。 |
| **generate** | **生成** | "Generate a pattern" = 生成一个图案。 |
| **return** | **返回** | 函数执行后需要输出的结果。**极其重要**。 |
| **print / output** | **打印 / 输出** | 将结果显示在屏幕上。注意与 `return` 的区别。 |
| **modify / update** | **修改 / 更新** | 改变一个变量或数据结构的值。 |
| **in-place** | **原地（修改）** | 直接修改传入的列表本身，而不是创建一个新的列表副本。 |
| **reverse** | **反转 / 颠倒** | 将序列的顺序倒过来，如 `[1,2,3]` -> `[3,2,1]`。 |
| **rotate** | **旋转** | 将列表的元素循环移动。 |
| **swap** | **交换** | 交换两个变量或元素的位置。 |
| **flatten** | **展开 / 扁平化** | 将嵌套列表变成一个一维列表。 |
| **concatenate** | **连接 / 拼接** | 将两个字符串或列表首尾相连。 |
| **parse** | **解析** | 从一个字符串中提取有用的信息。 |
| **round** | **四舍五入** | "Rounded to two decimal places" = 四舍五-入到两位小数。 |
| **sort** | **排序** | 按特定顺序（升序/降序）排列元素。 |
| **ascending (ASC)** | **升序** | 从小到大。 |
| **descending (DESC)** | **降序** | 从大到小。 |
| **initialize** | **初始化** | 为变量赋予一个初始值。 |
| **iterate** | **迭代** | 重复执行一个过程，通常指循环。 |
| **define** | **定义** | "Define a function" = 定义一个函数。 |
| **assign** | **赋值** | `x = 5` 就是一个赋值操作。 |
| **assume** | **假设** | "Assume the input is always valid" = 假设输入总是有效的。 |

---
### **三、 条件、逻辑与关系 (Conditions, Logic & Relations)**

| 英文单词 | 中文释义 | 常见语境与说明 |
| :--- | :--- | :--- |
| **constraint / restriction / rule** | **约束 / 限制 / 规则** | 题目对输入、输出或逻辑的限制。 |
| **valid / invalid** | **有效的 / 无效的** | "Valid input" = 有效的输入。 |
| **arbitrary** | **任意的** | "A list of arbitrary depth" = 任意深度的嵌套列表。 |
| **adjacent** | **相邻的** | "Adjacent blocks" = 相邻的方块。 |
| **respectively** | **分别地，依次地** | A and B are 1 and 2, respectively. (A是1, B是2)。 |
| **inclusive** | **包含（边界）的** | "The range [l, r] inclusive" = 包含 l 和 r。 |
| **exclusive** | **不包含（边界）的** | "The range [l, r) exclusive" = 包含 l 但不包含 r。 |
| **consecutive / continuous** | **连续的** | "Three consecutive identical blocks" = 三个连续相同的方块。 |
| **substring** | **子字符串** | 字符串中连续的一部分。 |
| **palindrome** | **回文** | 正读和反读都一样的字符串，如 "level"。 |
| **duplicate** | **重复的** | "Remove duplicate elements" = 移除重复元素。 |
| **unique** | **唯一的** | "Count unique words" = 统计不同单词的数量。 |
| **if tied...** | **如果（结果）相同...** | 规定了当出现多个最优解时的处理方式，如 "return the smallest index"。 |
| **approach** | **方法，途径** | 解决问题的思路或方法。 |
| **method** | **方法** | 除了指函数，也指解决问题的具体步骤。 |
| **concept** | **概念** | 一个抽象的想法，如“递归”是一个概念。 |
| **context** | **上下文，环境** | 一个词或事件所处的背景。 |
| **distribute** | **分布，分配** | |
| **establish** | **建立** | "Establish a connection" = 建立一个连接。 |
| **estimate** | **估计** | "Estimate the result" = 估计结果。 |
| **evidence** | **证据** | "Linguistic evidence" = 语言学证据。 |
| **factor** | **因素，因子** | 影响结果的原因之一。 |
| **indicate** | **表明，指出** | "The result indicates that..." = 结果表明... |
| **individual** | **单独的，个别的** | "Solve individually" = 独立解决。 |
| **involve** | **涉及，包含** | "The process involves three steps" = 该过程包含三个步骤。 |
| **issue** | **问题** | 需要解决的难题。 |
| **occur** | **发生** | "An error occurred" = 发生了一个错误。 |
| **process** | **过程，处理** | "The data processing pipeline" = 数据处理流水线。 |
| **require** | **要求** | "The function requires two parameters" = 该函数要求两个参数。 |
| **respond** | **响应，回答** | 程序对输入的反应。 |
| **source** | **来源，源头** | "Source code" = 源代码。 |
| **specific** | **特定的** | "A specific format" = 一种特定的格式。 |
| **similar** | **相似的** | "The two algorithms are similar" = 这两个算法是相似的。 |
| **consist of** | **由...组成** | "The list consists of integers" = 该列表由整数组成。 |

---
### **四、 输入/输出格式 (I/O Formatting)**

| 英文单词 | 中文释义 | 常见语境与说明 |
| :--- | :--- | :--- |
| **input** | **输入** | 程序需要读取的数据。 |
| **output** | **输出** | 程序需要打印的结果。 |
| **format** | **格式** | 规定了输入输出的样式。 |
| **specification** | **规范，说明** | 对任务或格式的详细描述。 |
| **line** | **行** | "The first line contains an integer n" = 第一行包含一个整数n。 |
| **separated by (spaces/tabs)** | **以（空格/制表符）分隔** | "A line of integers separated by spaces" = 一行由空格隔开的整数。 |
| **decimal places** | **小数位数** | "Round to two decimal places" = 保留两位小数。 |
| **prepend** | **前置，在前面加上** | "Prepend a space" = 在前面加一个空格。 |
| **append** | **追加，在末尾加上** | "Append a character to the string" = 在字符串末尾追加一个字符。 |
| **trailing (spaces/characters)** | **末尾的（空格/字符）** | 指一行内容结束后的多余字符，通常要求没有。 |
| **case-sensitive** | **大小写敏感** | `A` 和 `a` 被视为不同的字符。 |
| **case-insensitive** | **大小写不敏感** | `A` 和 `a` 被视为相同的字符。 |