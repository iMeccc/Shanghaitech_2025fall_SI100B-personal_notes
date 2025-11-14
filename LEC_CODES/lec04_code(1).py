###################
# EXAMPLE: floating point
################### 
# x = 0
# for i in range(10):
#     x += 0.1
# print(x == 1)
# print(x, 'is the same as?', 10*0.1)


###################
## EXAMPLE: Convert to binary
## use Python Tutor to go step-by-step: http://pythontutor.com/
###################

## Only positive numbers
num = 1507
result = ''
if num == 0:
    result = '0'
while num > 0:
    result = str(num%2) + result
    num = num//2
print(result)

## Can handle negative numbers
# num = -15
# if num < 0:
#     is_neg = True
#     num = abs(num)
# else:
#     is_neg = False
# result = ''
# if num == 0:
#     result = '0'
# while num > 0:
#     result = str(num%2) + result
#     num = num//2
# if is_neg:
#     result = '-' + result



#############
## EXAMPLE
# protip: use Python Tutor to go step-by-step: http://pythontutor.com/
#############

x = float(input('Enter a decimal number between 0 and 1: '))

p = 0
while ((2**p)*x)%1 != 0:
    print(f'Remainder = {str((2**p)*x - int((2**p)*x))}')
    p += 1

num = int(x*(2**p))

result = ''
if num == 0:
    result = '0'
while num > 0:
    result = str(num%2) + result
    num = num//2

for i in range(p - len(result)):
    result = '0' + result

result = result[0:-p] + '.' + result[-p:]
print(f'The binary representation of the decimal {str(x)} is {str(result)}')


#################
## EXAMPLE: successive addition
#################

x = 0
for i in range(10):
    x += 0.125
print(x == 1.25)

#######

x = 0
for i in range(10):
   x += 0.1
print(x == 1)

print(x, '==', 10*0.1)



####################################################
##################### AT HOME ######################
######################################################
# Write code that counts how many unique common characters there are between 
# two strings. For example below, the common characters count is 8: 
# text1 = "may the fourth be with you"
# text2 = "revenge of the sixth"
# Hint, start to write your code with a smaller example, then test it on the above text.

# text1 = "abc"
# text2 = "cde"
# your code here

####################################################
##################### END AT HOME ######################
######################################################



########################################################
############# ANSWERS TO AT HOME #######################
#######################################################
# Write code that counts how many unique common characters there are between 
# two strings. For example below, the common characters count is 8: 
# text1 = "may the fourth be with you"
# text2 = "revenge of the sixth"
# Hint, write your code with a smaller example.

# text1 = "may the fourth be with you"
# text2 = "revenge of the sixth"
# count = 0
# already = ""
# for i in text1:
#     if i in text2 and i not in already:
#         count += 1
#         already += i
# print(count)

####################################################
##################### END ANSWERS TO AT HOME ######################
######################################################

######################################################
############# ANSWERS TO LECTURE #####################
######################################################
# You Try It 1:
# Write code that loops a for loop over some range 
# and prints how many even numbers are in that range. Try it with:
# range(5)
# range(10)
# range(2,9,3)
# range(-4,6,2)
# range(5,6)

# evens = 0
# for i in range(5):
#      if i % 2 == 0:
#          evens += 1
# print(evens)


# You Try It 2:
# Assume you are given a string of lowercase letters in variable s. 
# Count how many unique letters there are in s. For example, if 
# s = "abca" Then your code prints 3. 

# your code here
# s='abca'
# seen = ""
# for char in s:
#     if char not in seen:
#         seen += char
# print(len(seen))



# You Try It 3:
# Hardcode a number as a secret number. Write a program that 
# checks through all the numbers between 1 to 10 and prints the 
# secret value. If it's not found, it doesn't print anything. 

# your code here
# one way
# secret = 6
# for i in range(1, 11):
#     if i == secret:
#         print("found")

# another way
# secret = 6
# if secret in range(1, 11):
#     print("found")


# You Try It 4:
# Hardcode a number as a secret number. Write a program that 
# checks through all the numbers between 1 to 10 and prints the 
# secret value. If it's not found, prints that it didn't find it. 

# your code here
# one way
# secret = 7
# found_flag = False
# for i in range(1, 11):
#     if i == secret:
#         found_flag = True
#         print("found")
# if found_flag == False:
#     print("not found")


######################################################
############# END ANSWERS TO LECTURE #####################
######################################################


#################
## EXAMPLE: successive addition
#################

# 0.125 is a perfect power of 2
# x = 0
# for i in range(10):
#     x += 0.125
# print(x == 1.25)

#######

# 0.1 is not a perfect power of 2
# x = 0
# for i in range(10):
#     x += 0.1
# # print(x == 1)

# print(x, '==', 10*0.1)

#############
## EXAMPLE
# protip: use Python Tutor to go step-by-step: http://pythontutor.com/
#############

# x = float(input('Enter a decimal number between 0 and 1: '))

# p = 0
# while ((2**p)*x)%1 != 0:
#     print(f'Remainder = {str((2**p)*x - int((2**p)*x))}')
#     p += 1

# num = int(x*(2**p))

# result = ''
# if num == 0:
#     result = '0'
# while num > 0:
#     result = str(num%2) + result
#     num = num//2

# for i in range(p - len(result)):
#     result = '0' + result

# result = result[0:-p] + '.' + result[-p:]
# print(f'The binary representation of the decimal {str(x)} is {str(result)}')


################
## EXAMPLE: Approximation by epsilon increments
## Incrementally fixing code as we find issues with approximation
################

# try with 36, 24, 2, 12345
x = 24 # 54321
epsilon = 0.01
num_guesses = 0
guess = 0.0
increment = 0.0001
while abs(guess**2 - x) >= epsilon:
    guess += increment
    num_guesses += 1
print(f'num_guesses = {num_guesses}')
print(f'{guess} is close to square root of {x}')

###########

# Caution, you'll need to "Restart Kernel" in the shell if you run this code
# x = 54321
# epsilon = 0.01
# num_guesses = 0
# guess = 0.0
# increment = 0.0001
# while abs(guess**2 - x) >= epsilon:
#     guess += increment
#     num_guesses += 1
#     if num_guesses%100000 == 0:
#         print(f'Current guess = {guess}')
#         print(f'Current guess**2 - x = {abs(guess*guess - x)}')
#     if num_guesses%1000000 == 0:
#         input('continue?')
# print(f'num_guesses = {num_guesses}')
# print(f'{guess} is close to square root of {x}')

##########

# Add an extra stopping condition 
# and check for why the loop terminated
# x = 54321
# epsilon = 0.01
# num_guesses = 0
# guess = 0.0
# increment = 0.0001  # try with 0.00001
# while abs(guess**2 - x) >= epsilon and guess**2 <= x:
#     guess += increment
#     num_guesses += 1
# print(f'num_guesses = {num_guesses}')
# if abs(guess**2 - x) >= epsilon:
#     print(f'Failed on square root of {x}')
#     print(f'Last guess was {guess}')
#     print(f'Last guess squared is {guess*guess}')
# else:
#     print(f'{guess} is close to square root of {x}')
    
#######


#################################################
######################## AT HOME ##########################
#################################################
# 1. If you are incrementing from 0 by 0.022, how many increments 
# can you do before you get a floating point error? 

# x = 0
# count = 20     # check different numbers here
# for i in range(count):
#     x += 0.022 # increment
#     print(x)      # check this value for floating point error


# 2. Automate the code from the previous problem. Suppose you are 
# just given an increment value. Write code that automatically
# determines how many times you can add increment to itself 
# until you start to get a floating point error.

# your code here

#################################################
#################################################
#################################################


#################################################
################ ANSWER TO AT HOME ##########################
#################################################
# Automate the code. Suppose you are 
# just given an increment value. Write code that automatically
# determines how many times you can add increment to itself 
# until you start to get a floating point error.

# n = 0.022
# N = 1
# x = n
# while x == n*N:
#     print(x)
#     x += n
#     N += 1
# note that the x and N increments one extra time 
# print(f'count is {N-1} where {x-n} != {n*(N-1)}')

#################################################
#################################################
#################################################

#####################
## EXAMPLE: fast square root using bisection search
#####################

# x = 54321  # try 0.5
# epsilon = 0.01
# num_guesses = 0
# low = 0.0
# high = x
# guess = (high + low)/2

# while abs(guess**2 - x) >= epsilon:
#     # uncomment to see each step's guess, high, and low 
#     #print(f'low = {str(low)} high = {str(high)} guess = {str(guess)}')
#     if guess**2 < x:
#         low = guess
#     else:
#         high = guess
#     guess = (high + low)/2.0
#     num_guesses += 1
# print(f'num_guesses = {str(num_guesses)}')
# print(f'{str(guess)} is close to square root of {str(x)}')



############### YOU TRY IT ###################
# x = 0.5
# epsilon = 0.01
# # choose the low endpoint
# low = ???
# # choose the high endpopint
# high = ???

# guess = (high + low)/2

# while abs(guess**2 - x) >= epsilon:
#     #print(f'low = {str(low)} high = {str(high)} guess = {str(guess)}')
#     if guess**2 < x:
#         low = guess
#     else:
#         high = guess
#     guess = (high + low)/2.0
# print(f'{str(guess)} is close to square root of {str(x)}')

#####################################################


#####################
## Code for square root with all x values
#####################
#x = 0.5
#epsilon = 0.01
#if x >= 1:
#    low = 1.0
#    high = x
#else:
#    low = x
#    high = 1.0
#guess = (high + low)/2
#
#while abs(guess**2 - x) >= epsilon:
#    print(f'low = {str(low)} high {str(high)} guess = {str(guess)}')
#    if guess**2 < x:
#        low = guess
#    else:
#        high = guess
#    guess = (high + low)/2.0
#print(f'{str(guess)} is close to square root of {str(x)}')
