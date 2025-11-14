def isMatch(s: str, p: str) -> bool:
    # memo 用来存储 (i, j) -> bool 的计算结果
    memo = {}

    def dp(i, j):
        # 如果 (i, j) 已经计算过，直接返回结果
        if (i, j) in memo:
            return memo[(i, j)]

        # 基本情况：模式 p 已经用完
        if j == len(p):
            # 只有当 s 也用完时，才算匹配
            return i == len(s)

        # --- 递推逻辑 ---
        # 判断当前字符是否匹配
        first_match = (i < len(s)) and (p[j] == s[i] or p[j] == '.')

        # 情况一：p 的下一个字符是 '*'
        if j + 1 < len(p) and p[j+1] == '*':
            # 做出选择
            # choice1: 忽略 p[j]*，即匹配零次
            # choice2: p[j]* 匹配 s[i]，然后继续用 p[j:] 匹配 s[i+1:]
            result = dp(i, j + 2) or (first_match and dp(i + 1, j))
        
        # 情况二：p 的下一个字符不是 '*'
        else:
            result = first_match and dp(i + 1, j + 1)
        
        # 将结果存入备忘录
        memo[(i, j)] = result
        return result

    # 从头开始进行匹配
    return dp(0, 0)

char = input().strip().split(' ')
s = char[0]
p = char[1]    
print(isMatch(s,p))