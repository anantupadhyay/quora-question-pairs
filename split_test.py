import csv

test = open('test.csv', 'r').readlines()
fname = 1
for x in xrange(len(test)):
	if x%100000 == 0:
		open(str(fname) + '.csv', 'w+').writelines(test[x:x+100000])
		fname += 1