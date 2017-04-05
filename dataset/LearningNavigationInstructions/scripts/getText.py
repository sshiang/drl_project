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
import re

parser = argparse.ArgumentParser(description='MultiRank for object recognition improvement.')
parser.add_argument('-m', '--mapDir', dest='mapDir', default="../maps", help='Map directory')
parser.add_argument('-i', '--input', dest='input', default='../data/semantic/SingleSentences-sail.xml', help='input xml file')
parser.add_argument('-j', '--demo_input', dest='demo_input', default='../data/SingleSentence.xml', help='input xml file')
parser.add_argument('-s', '--source', dest='source', default='../data/sail_source', help='source states feature output')
parser.add_argument('-t', '--target', dest='target', default='../data/sail_target', help='target commands output')
parser.add_argument('-d', '--index', dest='index', default=None, help='shuffle index')
parser.add_argument('-o', '--output', dest='output', default="", help='shuffle index')
parser.add_argument('-c', '--cluster', dest='cluster', default=True, help='shuffle index')
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
	envs[name] = readMap("%s/%s"%(opts.mapDir, filename))
	

def getMapID(string):
	map_id = string.split("_")[1][:-1]
	return map_id

# read sail commands
tree = ET.parse(opts.input)
root = tree.getroot()


commands_id = {}
sail_commands_id = {}

commands = []
sail_commands = []
#f_tgt = open(opts.target,"w")
for examples in root:
	filename = examples.attrib["file"].lower()
	ID = examples.attrib['id'].lower()
	task_ID = ID.split("-")[0]
	print task_ID
	mapid = getMapID(examples.attrib["file"].lower())
	for example in examples:
		if example.tag == "nl":
			text = example.text.strip().lower()
			print text
			#f_tgt.write(example.text.strip()+"\n")
			commands.append(text)
			if task_ID not in commands_id:
				commands_id[task_ID] = [text]
			else:
				commands_id[task_ID].append(text)

		elif example.tag == "mrl":
			texts = [x.strip().replace(","," , ") for x in example.text.strip().lower().split("\n")]
			print texts
			sail_commands.append(texts)
			if task_ID not in sail_commands_id:
				sail_commands_id[task_ID] = []
			sail_commands_id[task_ID].append(texts)


new_commands = []
new_sail_commands = []

mapping = {}
for i in range(len(commands)):
	string = "%s %s"%(commands[i], " ".join(sail_commands[i]))
	if string not in mapping:
		mapping[string] = True
		new_commands.append(commands[i])
		new_sail_commands.append(sail_commands[i])
		
commands = new_commands
sail_commands = new_sail_commands

mapping = {}

new_commands_id = {}
new_sail_commands_id = {}
for ID in commands_id:
	new_commands_id[ID] = []
	new_sail_commands_id[ID] = []
	print commands_id[ID]
	print sail_commands_id[ID]
	print len(commands_id[ID]), len(sail_commands_id[ID])
	for i in range(len(commands_id[ID])):
		string = "%s %s"%(commands_id[ID][i], " ".join(sail_commands_id[ID][i]))
		#print string
		if string not in mapping:
			new_commands_id[ID].append(commands_id[ID][i])
			new_sail_commands_id[ID].append(sail_commands_id[ID][i])
			mapping[string] = True

commands_id = new_commands_id
sail_commands_id = new_sail_commands_id



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
	index_file = "sail_index.pkl"
	index_id_file = "sail_index_id.pkl"
else:
	index_file = opts.index
	index_id_file = opts.index.replace(".pkl","")+"_id.pkl"

if os.path.exists(index_file):
	indexes = loadPickle(index_file)
else:
	indexes = range(len(commands))
	random.shuffle(indexes)
	savePickle(index_file, indexes)

if os.path.exists(index_id_file):
	indexes_id = loadPickle(index_id_file)
else:
	indexes_id = commands_id.keys()
	random.shuffle(indexes_id)
	savePickle(index_id_file, indexes_id)

def dumpPickle(path, data):
	output = open(path, 'wb')
	cPickle.dump(data, output)
	output.close()

#f_tgt.close()
for i in range(len(ratios)):
	start = int(len(commands)*sum(ratios[:i]))
	end = int(len(commands)*sum(ratios[:i+1]))

	f = open("commands.set."+sets[i]+".txt","w")
	f2 = open("sail_commands.set."+sets[i]+".txt","w")
	for j in range(start,end):
		f.write(commands[indexes[j]]+"\n")
		for k in range(len(sail_commands[indexes[j]])):
			f2.write(sail_commands[indexes[j]][k].strip().replace("  "," ")+" ")
		f2.write("\n")
	f.close()
	f2.close()


count = 0
for i in range(len(ratios)):
	start = int(len(indexes_id)*sum(ratios[:i]))
	end = int(len(indexes_id)*sum(ratios[:i+1]))

	f = open("commands_id.set."+sets[i]+".txt","w")
	f2 = open("sail_commands_id.set."+sets[i]+".txt","w")
	f3 = open("cluster_mapping.set."+sets[i]+".txt","w")
	for j in range(start,end):
		count += 1
		for k in range(len(commands_id[indexes_id[j]])):
			f.write(commands_id[indexes_id[j]][k]+"\n")
			f3.write(indexes_id[j]+"\n")
			for l in range(len(sail_commands_id[indexes_id[j]][k])):
				f2.write(sail_commands_id[indexes_id[j]][k][l].strip().replace("  "," ")+" ")
			f2.write("\n")
	f.close()
	f2.close()
	f3.close()

