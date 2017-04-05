import sys
import os 


import argparse
parser = argparse.ArgumentParser(description='MultiRank for object recognition improvement.')
parser.add_argument('-c', '--cluster', dest='cluster', default="cluster.mapping.txt", help='cluster mapping')
parser.add_argument('-i', '--input', dest='input', default="translation.txt", help='translation output')
parser.add_argument('-o', '--output', dest='output', default="translation.aggregate.txt", help='aggregate output')
opts = parser.parse_args()


def readResult(path):
	results = []
	for line in open(path):
		results.append(line.strip())
	return results


translation = readResult(opts.input)

f = open(opts.output,'w')
ID = ""
count = 0
for line in open(opts.cluster):
	now_ID = line.strip()
	if now_ID != ID and ID!="":
		f.write("\n"+translation[count]+" ")
	else:
		f.write(translation[count]+" ")
	ID = now_ID
	count += 1
f.write("\n")
f.close()
