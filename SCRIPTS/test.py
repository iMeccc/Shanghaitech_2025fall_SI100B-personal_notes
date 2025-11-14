n = int(input())
def countstep(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    elif n>2:
        return countstep(n-1)+countstep(n-2)
print(countstep(n))