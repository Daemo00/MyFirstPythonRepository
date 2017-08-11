"""Take two lists, say for example these two:
	a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
	b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
and write a program that returns a list that contains only the elements that are common between the lists (without duplicates). Make sure your program works on two lists of different sizes. Write this using at least one list comprehension. (Hint: Remember list comprehensions from Exercise 7).

The original formulation of this exercise said to write the solution using one line of Python, but a few readers pointed out that this was impossible to do without using sets that I had not yet discussed on the blog, so you can either choose to use the original directive and read about the set command in Python 3.3, or try to implement this on your own and use at least one list comprehension in the solution.

Extra:
Randomly generate two lists to test this
"""
import random

        
a = [random.randint(0, 100) for i in range(random.randint(0, 20))]
b = [random.randint(0, 100) for i in range(random.randint(0, 20))]

print("a = {}".format(a))
print("b = {}".format(b))

a_and_b = [x for x in a if x in b]

res = []
while len(a_and_b) > 0:
    el = a_and_b.pop()
    if el not in res:
        res.append(el)

print(res)