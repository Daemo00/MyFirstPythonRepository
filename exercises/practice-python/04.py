"""Create a program that asks the user for a number and then prints out a list of all the divisors of that number. (If you don't know what a divisor is, it is a number that divides evenly into another number. For example, 13 is a divisor of 26 because 26 / 13 has no remainder.)"""
from math import floor, sqrt

n = int(input("Choose a number "))

print(sorted(set([i for sub in [(m, n // m) for m in range(1, int(floor(sqrt(n))) + 1) if n % m == 0] for i in sub ])))
