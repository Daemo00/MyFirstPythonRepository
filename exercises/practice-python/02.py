"""Ask the user for a number. Depending on whether the number is even or odd, print out an appropriate message to the user. Hint: how does an even / odd number react differently when divided by 2?

Extras:

If the number is a multiple of 4, print out a different message.
Ask the user for two numbers: one number to check (call it num) and one number to divide by (check). If check divides evenly into num, tell that to the user. If not, print a different appropriate message."""

n = int(input("Enter a number to check: "))
m = int(input("Enter a number to divide by: "))

if n % 4 == 0:
    print("{} is {} times 4".format(n, n // 4))
elif n % 2 == 0:
    print("{} is {} times 2".format(n, n // 2))
else:
    print("{} is odd".format(n))

if n % m == 0:
    print("{} is {} times {}".format(n, n // m, m))
else:
    print("{} is not evenly divided by {}".format(n, m))
