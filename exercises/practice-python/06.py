"""Ask the user for a string and print out whether this string is a palindrome or not. (A palindrome is a string that reads the same forwards and backwards.)"""

word = input("Give me a word ")

if word == word[::-1]:
    print("It is a palindrome!")
else:
    print("It is not a palindrome")
