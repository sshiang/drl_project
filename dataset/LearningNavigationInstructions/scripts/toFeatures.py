import xml.etree.ElementTree as ET
import ast
import argparse
import heapq
from collections import namedtuple
import os
import sys
import re
import cPickle
import math
import random 

parser = argparse.ArgumentParser(description='MultiRank for object recognition improvement.')
parser.add_argument('-m', '--mapDir', dest='mapDir', default="../maps", help='Map directory')
parser.add_argument('-i', '--input', dest='input', default='../data/SingleSentence.xml', help='input xml file')
parser.add_argument('-s', '--source', dest='source', default='../data/source', help='source states feature output')
parser.add_argument('-t', '--target', dest='target', default='../data/target', help='target commands output')
parser.add_argument('-d', '--index', dest='index', default=None, help='shuffle index')
parser.add_argument('-o', '--output', dest='output', default="", help='shuffle index')
parser.add_argument('-p', '--process', dest='process', default="", help='shuffle index')
opts = parser.parse_args()

ratios = [0.7,0.1,0.2]
sets = ['train','valid','test']

class Maps:
	def __init__(self, name="", nodes={}, edges={}):
		self.name=name
		self.edges=edges
		self.nodes=nodes

def readMap(path):
	global wallMap 
	global floorMap 
	global itemMap

	tree = ET.parse(path)
	root = tree.getroot()
	
	name = path.split("/")[-1].replace(".xml","")

	nodes = {}
	edges = {}
	for child in root:
		for c in child:
			if c.tag == "node":
				(x,y) = (int(c.attrib['x']), int(c.attrib['y']))
				obj = c.attrib['item']
				itemMap[obj] = itemMap.get(obj,len(itemMap))
				nodes[(x,y)] = obj	
			elif c.tag == "edge":
				(x1,y1) = [int(x) for x in c.attrib['node1'].split(',')]
				(x2,y2) = [int(x) for x in c.attrib['node2'].split(',')]
				wall = c.attrib['wall']
				floor = c.attrib['floor']
				wallMap[wall] = wallMap.get(wall,len(wallMap))
				floorMap[floor] = floorMap.get(floor,len(floorMap))

				if (x1,y1) not in edges:
					edges[(x1,y1)] = {(x2,y2):(wall,floor)}
				else:
					edges[(x1,y1)][(x2,y2)] = (wall,floor)

				if (x2,y2) not in edges:
					edges[(x2,y2)] = {(x1,y1):(wall,floor)}
				else:
					edges[(x2,y2)][(x1,y1)] = (wall,floor)

	return Maps(name,nodes,edges)

def getShapeIndex(a,b,c,d):
	x = [a,b,c,d]
	return sum([x[i]*(2**(i)) for i in range(len(x))])	

def getFeatures(paths, i, env):
	global wallMap 
	global floorMap 
	global itemMap

	# is_start
	# angle
	# difference in angle
	# abs(diff(angle))
	# difference in x
	# difference in y
	# abs(diff(x))
	# abs(diff(y))
	# wall (t-1)
	# floor (t-1)
	# item (t-1)
	node, fea = paths[i], []
	# is start
	fea+= [int(node[-1]==-1)]
	# angle, diff(angle), abs(diff(angle))
	if i == 0:
		fea+=[-1, 0, 0]
	else:
		diff = paths[i][-1]-paths[i-1][-1]
		fea+=[paths[i][-1], diff, math.fabs(diff)]
	# diff(x,y), abs(diff(x,y))
	if i == 0:
		fea += [0,0,0,0]
	else:
		delta_x = paths[i][0]-paths[i-1][0]
		delta_y = paths[i][1]-paths[i-1][1]
		fea += [delta_x, delta_y, math.fabs(delta_x), math.fabs(delta_y)]
	# wall
	f_wall = [0 for x in range(len(wallMap))]
	f_floor = [0 for x in range(len(floorMap))]
	f_item = [0 for x in range(len(itemMap))]
	obj = env.nodes[(node[0],node[1])]

	if obj != "":
		f_item[itemMap[obj]] = 1

	if i != 0 and paths[i-1][:2]!=paths[i][:2]:
			
		(wall, floor) = env.edges[(node[0],node[1])][(paths[i-1][0],paths[i-1][1])]
		if wall!="":
			f_wall[wallMap[wall]] = 1
		if floor!="":
			f_floor[floorMap[floor]] = 1

	fea += f_wall
	fea += f_floor
	fea += f_item


	# SHAPE
	f_shape = [0 for x in range(2**(4))]
	pos_x = paths[i][0]
	pos_y = paths[i][1]
	left = 1 if (pos_x-1, pos_y) in env.edges[(pos_x, pos_y)] else 0
	right = 1 if (pos_x+1, pos_y) in env.edges[(pos_x, pos_y)] else 0
	up = 1 if (pos_x, pos_y+1) in env.edges[(pos_x, pos_y)] else 0
	down = 1 if (pos_x, pos_y-1) in env.edges[(pos_x, pos_y)] else 0
	f_shape[getShapeIndex(left,right,up,down)] = 1
	fea += f_shape
	# INTERSECT
	inters = sum([left, right, up, down])
	f_inters = [0 for x in range(4)]
	f_inters[inters-1] = 1
	fea += f_inters
	return fea

wallMap = {}
floorMap = {}
itemMap = {}

envs = {}
for filename in os.listdir(opts.mapDir):
	if ".xml" not in filename or filename[0]==".":
		continue
	name = filename.replace(".xml","").replace("map-","")
	print "%s/%s"%(opts.mapDir, filename)
	envs[name] = readMap("%s/%s"%(opts.mapDir, filename))
	

# read demonstrations and commands

tree = ET.parse(opts.input)
root = tree.getroot()

commands = []
features = []
#f_tgt = open(opts.target,"w")
for examples in root:
	mapid = examples.attrib["map"].lower()
	for example in examples:
		if example.tag == "instruction":
			#f_tgt.write(example.text.strip()+"\n")
			commands.append(example.text.strip().lower())
		elif example.tag == "path":
			feature = []
			paths = [[int(y) for y in x.split(",")] for x in re.findall(r'\(\s*([^"]*?)\s*\)', example.text)]				
			for i in range(len(paths)):
				feature.append(getFeatures(paths, i, envs[mapid]))
			features.append(feature)

# clean duplicate
memory_features = {}
for i in range(len(commands)):
	if commands[i] not in memory_features:
		memory_features[str(commands[i])] = [features[i]]
	else:
		if features[i] in memory_features[str(commands[i])]:
			continue
		memory_features[str(commands[i])].append(features[i])

features = []
commands = []
for key in memory_features:
	for fea in memory_features[key]:
		features.append(fea)
		commands.append(key)

def loadPickle(path):
	fi = open(path,'rb')
	data = cPickle.load(fi)
	fi.close()
	return data

def savePickle(path, data):
	fi = open(path,'wb')
	cPickle.dump(data, fi)
	fi.close()
	
if opts.index == None:
	index_file = "index.pkl"
else:
	index_file = opts.index

if os.path.exists(index_file):
	indexes = loadPickle(index_file)
else:
	indexes = range(len(features))
	random.shuffle(indexes)
	savePickle('index.pkl', indexes)


#f_tgt.close()
for i in range(len(ratios)):
	start = int(len(features)*sum(ratios[:i]))
	end = int(len(features)*sum(ratios[:i+1]))
	features_set = [features[indexes[x]] for x in range(start,end)]

	f = open(opts.target+"."+sets[i]+".txt","w")
	for j in range(start,end):
		f.write("<s> "+commands[indexes[j]]+" </s>\n")
	f.close()

	output = open(opts.source+"."+sets[i]+".pkl", 'wb')
	cPickle.dump(features_set, output)
	output.close()


print wallMap
print floorMap
print itemMap 
