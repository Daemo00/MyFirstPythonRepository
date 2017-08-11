"""Create a program that asks the user to enter their name and their age. Print out a message addressed to them that tells them the year that they will turn 100 years old.

Extras:

Add on to the previous program by asking the user for another number and printing out that many copies of the previous message. (Hint: order of operations exists in Python)
Print out that many copies of the previous message on separate lines. (Hint: the string "\n is the same as pressing the ENTER button)"""
from datetime import date


name = input("What is your name? ")
age = int(input("What is your age? "))
times = int(input("How many times do I have to print it? "))

print( str("Hi %s, you will turn 100 in %d\n" % (name, date.today().year - age + 100))*times)
