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
	envs[name] = readMap("%s/%s"%(opts.mapDir, filename))
	

def getMapID(string):
	map_id = string.split("_")[1][:-1]
	return map_id

# read sail commands
tree = ET.parse(opts.input)
root = tree.getroot()

commands_map = {}
sail_commands_map = {}
#f_tgt = open(opts.target,"w")
for examples in root:
	filename = examples.attrib["file"].lower()
	ID = examples.attrib['id'].lower()
	mapid = getMapID(examples.attrib["file"].lower())
	for example in examples:
		if example.tag == "nl":
			#f_tgt.write(example.text.strip()+"\n")
			commands_map[filename+ID] = example.text.strip().lower()
		elif example.tag == "mrl":
			sail_commands_map[filename+ID] = example.text.strip().lower()

# read demonstrations and commands
tree = ET.parse(opts.demo_input)
root = tree.getroot()

paths_map = {}
features_map = {}
filename = ""
#f_tgt = open(opts.target,"w")
for examples in root:
	mapid = examples.attrib["map"].lower()
	ID = examples.attrib["id"].lower()
	for example in examples:
		if example.tag == "path":
			feature = []
			path = [[int(y) for y in x.split(",")] for x in re.findall(r'\(\s*([^"]*?)\s*\)', example.text)]				
			for i in range(len(path)):
				feature.append(getFeatures(path, i, envs[mapid]))
			paths_map[filename+ID] = path
			features_map[filename+ID] = feature
		elif example.tag == "instruction":
			filename = example.attrib['filename'].lower()


features = []
paths = []
commands = []
sail_commands = []
c = 0
for key in sail_commands_map:
	if key not in features_map:
		c += 1
		continue

	features.append(features_map[key])
	paths.append(paths_map[key])
	commands.append(commands_map[key])
	sail_commands.append(sail_commands_map[key])


'''
# clean duplicate
memory_commands = {}
for i in range(len(commands)):
	if commands[i] not in memory_commands:
		memory_commands[str(commands[i])] = [sail_commands[i]+p]
	else:
		if sail_commands[i] in memory_commands[str(commands[i])]:
			continue
		memory_commands[str(commands[i])].append(sail_commands[i])


sail_commands = []
commands = []
for key in memory_commands:
	for com in memory_commands[key]:
		sail_commands.append(com)
		commands.append(key)
	
'''

r1 = re.compile("(.*?)\s*\(")
r2 = re.compile(".*?\((.*?)\)")
	
nDims = 0
sail_map = {}
sail_process_commands = []

for i in range(len(sail_commands)):
	process = []
	for line in sail_commands[i].split("\n"):
		line = line.strip()

		if line == "null":
			action = "null"
			contexts = ""
		else:
			action = (r1.match(line)).group(1)
			contexts = (r2.match(line)).group(1)	

		#print len(contexts.split(","))
		for context in contexts.split(","):
			context = context.strip()
			pair = context.split(":")
			key = pair[0]
			value = ""

			if len(pair) > 1:
				value = pair[1]

			if action not in sail_map:
				sail_map[action] = {key:{value:nDims}}
				nDims += 1
			elif key not in sail_map[action]:
				sail_map[action][key] = {value:nDims}
				nDims += 1
			elif value not in sail_map[action][key]: 
				sail_map[action][key][value] = nDims
				nDims += 1

			process.append((action, key, value))

	sail_process_commands.append(process)

print sail_map

map_features = []
for i in range(len(sail_process_commands)):
	fea = [0 for x in range(nDims)]
	for s in sail_process_commands[i]:
		fea[sail_map[s[0]][s[1]][s[2]]] = 1
	map_features.append(fea)

print len(map_features)

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
else:
	index_file = opts.index

if os.path.exists(index_file):
	indexes = loadPickle(index_file)
else:
	indexes = range(len(features))
	random.shuffle(indexes)
	savePickle('sail_index.pkl', indexes)


def dumpPickle(path, data):
	output = open(path, 'wb')
	cPickle.dump(data, output)
	output.close()

#f_tgt.close()
for i in range(len(ratios)):
	start = int(len(features)*sum(ratios[:i]))
	end = int(len(features)*sum(ratios[:i+1]))
	paths_set = [paths[indexes[x]] for x in range(start, end)]
	sail_set = [sail_process_commands[indexes[x]] for x in range(start, end)]
	features_set = [features[indexes[x]] for x in range(start, end)]
	map_features_set = [map_features[indexes[x]] for x in range(start, end)]

	f = open(opts.target+"."+sets[i]+".txt","w")
	for j in range(start,end):
		f.write("<s> "+commands[indexes[j]]+" </s>\n")
	f.close()

	dumpPickle(opts.source+".path."+sets[i]+".pkl", paths_set)
	dumpPickle(opts.source+".sail."+sets[i]+".pkl", sail_set)
	dumpPickle(opts.source+".feature."+sets[i]+".pkl", features_set)
	dumpPickle(opts.source+".outFeature."+sets[i]+".pkl", map_features_set)
