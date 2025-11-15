def josephus(n, k):
    """
    模拟击鼓传花（约瑟夫环）问题
    :param n: 总人数
    :param k: 每次数k次淘汰一个人
    :return: 淘汰顺序列表
    """
    people = list(range(1, n + 1))  # 初始化参与人员（1到n编号）
    eliminated = []  # 记录淘汰顺序
    current = 0  # 当前开始数数的位置
    
    while people:
        # 计算要淘汰的位置：当前位置 + k-1，对剩余人数取模
        index = (current + k - 1) % len(people)
        # 移除并记录被淘汰的人
        eliminated.append(people.pop(index))
        # 更新下一次开始数数的位置（被淘汰者的下一个位置）
        current = index % len(people) if people else 0  # 若列表为空，current无效
    
    return eliminated


# 测试示例
if __name__ == "__main__":
    n = int(input("请输入人数 n："))
    k = int(input("请输入每次数的次数 k："))
    
    order = josephus(n, k)
    print(f"淘汰顺序为：{order}")