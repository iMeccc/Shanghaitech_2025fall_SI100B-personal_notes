n,k = map(int, input().strip().split())
num_list = list(range(1,n+1))
pointer = 0
out_list = []

def output(num_list,out_list,pointer,k):
    if num_list:
        pointer += k
        pointer = pointer%len(num_list)
        if pointer == 0:
            pointer = len(num_list)
        new_list = []
        for i in range(len(num_list)):
            if i == pointer-1:
                out_list.append(num_list[i])
            else:
                new_list.append(num_list[i])
        pointer -= 1
        out_list.extend(output(new_list,out_list,pointer,k))
        return out_list
    else:
        return []        
    
out_line = output(num_list,out_list,pointer,k)[0:n]

print(out_line)