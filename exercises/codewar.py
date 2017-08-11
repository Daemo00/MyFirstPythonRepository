def snail(array):
	res = array[0] \
			+ [x for x in [r[-1] for r in array]][1:] \
			+ array[-1][:-1][::-1] \
			+ [x for x in [r[0] for r in array]][1:-1][::-1]
	res = res + snail([x for x in [r[1:-1] for r in array]][1:-1])
	return res 

print(snail([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
##print(snail([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))