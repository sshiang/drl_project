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
parser.add_argument('-i', '--input', dest='input', default='../data/semantic/', help='input xml file')
parser.add_argument('-j', '--demo_input', dest='demo_input', default='../data/SingleSentence.xml', help='input xml file')
parser.add_argument('-s', '--source', dest='source', default='../data/sail_source', help='source states feature output')
parser.add_argument('-t', '--target', dest='target', default='../data/sail_target', help='target commands output')
parser.add_argument('-d', '--index', dest='index', default=None, help='shuffle index')
parser.add_argument('-o', '--output', dest='output', default="", help='shuffle index')
parser.add_argument('-p', '--process', dest='process', default="", help='shuffle index')
parser.add_argument('-c', '--clear', dest='clear', type=bool, default=False, help='shuffle index')
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


def findBracketFront(s, brackets):
	for i in range(len(s)):
		if s[i] in brackets:
			return i
	return -1

def findBracketBack(s, br):
	for i in range(len(s)-1, -1, -1):
		if s[i] == br:
			return i
	return -1

def getAttributes(s, brackets, brackets_rev):
	attrs = []
	brs = []
	start = 0
	for i in range(len(s)):
		if s[i] in brackets:
			brs.append(s[i])
		elif s[i] in brackets_rev: 
			if brs[-1] == brackets_rev[s[i]]:
				brs = brs[:-1]
		if len(brs)==0 and i != 0 and s[i] ==",":
			# compelete a segment 
			attrs.append(s[start:i].strip())
			start = i+1
	attrs.append(s[start:].strip())
	return attrs
	
def getCASFeatures(cas):

	fea = []

	#number of key information to remember  ???
	fea += [cas.count('=')]

	#low-level command groundtruth
	fea += [cas.count("(")]

	#CAS command maximum depth
	depth = 0
	brs = ""
	brackets = {"[":"]", "(":")"}
	for i in range(len(cas)):
		if cas[i] in brackets:
			brs += cas[i]
			depth = max(depth, len(brs)) #sum([brs.count(x) for x in brackets]))
		if cas[i] in brackets.values():
			brs = brs[:-1]
	fea += [depth]

	#number of defined attributes
	fea += [cas.count('=')]

	floors = ["honeycomb", "cement", "brick", "grass", "rose", "wood", "stone", "bluetile"]
	walls = ["fish", "butterfly", "eiffel"]

	#number of floor colors mentioned
	fea += [sum([cas.lower().count(x) for x in floors])]

	#number of wall colors mentioned
	fea += [sum([cas.lower().count(x) for x in walls])]

	#whether or not to head towards an object
	target = "thing("
	fea += [0]
	for i in range(len(cas)):
		if cas[i:i+len(target)].lower() == target:
			start, depth, subcas = i, "", ""
			for j in range(i, len(cas)):
				if cas == "(":
					subcas += cas[j]
				if cas[j] in brackets:
					depth += cas[j]
				if cas[j] in brackets.keys():
					depth = depth[:-1]
			if "front" in subcas.lower() and 'obj' in subcas.lower():
				fea[-1]+=1

	#number of landmarks mentioned
	fea += [cas.lower().count("thing(")]

	#turn reference frame
	fea += [cas.lower().count('turn')]

	return fea
	

def getStateFeatures(paths, idx , env):
	global wallMap 
	global floorMap 
	global itemMap

	fea = []
	changeOrientation = False
	#change orientation
	fea += [0]
	for i in range(len(paths[idx])):
		delta_a = paths[idx][i][-1]-paths[idx][i-1][-1] if i!=0 else 0
		if delta_a !=0:
			fea[-1] = 1
			changeOrientation = True
			break

	#change position
	fea += [0]
	if idx != len(paths):
		fea[-1] = 1

	#change orientation and then position 
	fea += [0]
	if changeOrientation==True and idx != len(paths):
		fea[-1] = 1 

	# change position and then orientation
	fea += [0]
	if changeOrientation==True and idx != 0:
		fea[-1] = 1

	#the final place contains an object 
	[x,y,a] = paths[idx][-1]	
	if (x,y) in env.nodes and env.nodes[(x,y)]!="":
		fea += [1]	
	else:
		fea += [0]

	#pass an object while walking
	# ---> walk straightly and then 
	fea += [0]
	if (x,y) in env.nodes and env.nodes[(x,y)]!="":
		if idx != 0 and idx != len(paths)-1:
			delta_x1, delta_y1 = paths[idx][0][0]-paths[idx-1][0][0], paths[idx][0][1]-paths[idx-1][0][1]
			delta_x2, delta_y2 = paths[idx+1][0][0]-paths[idx][0][0], paths[idx+1][0][1]-paths[idx][0][1]
			if delta_x1 == delta_x2 and delta_y1 == delta_y2:
				fea[-1] == 1

	#the final place is a dead-end
	fea += [1]
	[x,y,a] = paths[-1][-1]
	if (x,y) in env.edges: 
		if len(env.edges[x,y])<=1:
			fea[-1]=1
	
	#the final pose is the goal pose
	fea += [0]
	if idx == len(paths)-1:
		fea[-1]=1

	#it is the first action to take
	if paths[idx][0][-1] == -1:
		fea += [1]
	else:
		fea += [0]

	#final pose faces a new floor/wall color
	fea += [0, 0]
	[x,y,a] = paths[idx][-1]
	[x_p,y_p,a_p] = paths[idx-1][-1]
	if (x,y) in env.edges and (x_p!=x or y_p!=y) and idx != 0:
		wall_next, wall_this, floor_next, floor_this = "", "", "", ""
		if a == 0:
			if (x,y-1) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x,y-1)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]
		elif a == 90:
			if (x-1,y) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x-1,y)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]
		elif a == 180:
			if (x,y+1) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x,y+1)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]
		elif a == 270:
			if (x+1,y) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x+1,y)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]		

		if wall_next != "" and wall_this!=wall_next:
			fea[-2]=1
		if floor_next != "" and floor_this != floor_next:
			fea[-1]=1
 
	#an object is visible from the final pose 
	[x,y,a] = paths[idx][-1]
	fea += [0]
	if a == 0:
		while True:
			if (x,y-1) in env.nodes:
				if env.nodes[(x,y-1)] != "":
					fea[-1] = 1
			else:
				break
			y -= 1

	elif a == 90:
		while True:
			if (x-1,y) in env.nodes:
				if env.nodes[(x-1,y)] != "":
					fea[-1] = 1
			else:
				break
			x -= 1

			
	elif a == 180:
		while True:
			if (x,y+1) in env.nodes:
				if env.nodes[(x,y+1)] != "":
					fea[-1] = 1
			else:
				break
			y += 1


	elif a == 270:
		while True:
			if (x+1,y) in env.nodes:
				if env.nodes[(x+1,y)] != "":
					fea[-1] = 1
			else:
				break
			x += 1


	#there is an object at the turn 
	fea += [0]
	if len(paths[idx])!=1 and (x,y) in env.nodes and env.nodes[(x,y)] != "":
		fea[-1]=1

	
	#the place where to turn at is a dead-end
	fea += [0]
	if len(paths[idx])>1: # is as turn 
		[x,y,a] = paths[idx][-1]
		if a == 0:
			while True:
				if (x,y-1) in env.nodes:
					length = len(env.edges[(x,y-1)])
					if length ==1:
						fea[-1] = 1
						break
					elif length >2:
						break
				else:
					break
				y -= 1
		elif a == 90:
			while True:
				if (x-1,y) in env.nodes:
					length = len(env.edges[(x-1,y)])
					if length ==1:
						fea[-1] = 1
						break
					elif length >2:
						break
				else:
					break
				x-=1
		elif a == 180:
			while True:
				if (x,y+1) in env.nodes:
					length = len(env.edges[(x,y+1)])
					if length ==1:
						fea[-1] = 1
						break
					elif length >2:
						break
				else:
					break
				y+=1
		elif a == 270: 				
			while True:
				if (x+1,y) in env.nodes:
					length = len(env.edges[(x+1,y)])
					if length ==1:
						fea[-1] = 1
						break
					elif length >2:
						break
				else:
					break
				x+=1

	return fea



def getPaperFeatures(paths, env):
	global wallMap 
	global floorMap 
	global itemMap

	fea = []

	#change orientation
	fea += [0]
	for i in range(len(paths)):
		delta_a = paths[i][-1]-paths[i-1][-1] if i!=0 else 0
		if delta_a !=0:
			fea[-1] = 1
			break

	#change position

	fea += [0]
	for i in range(len(paths)):
		delta_x = paths[i][0]-paths[i-1][0] if i!=0 else 0
		delta_y = paths[i][1]-paths[i-1][1] if i!=0 else 0
		if delta_x !=0 or delta_y != 0:
			fea[-1] = 1
			break

	#change orientation and then position 
	changeFlag = False
	fea += [0]
	for i in range(len(paths)):
		delta_x = paths[i][0]-paths[i-1][0] if i!=0 else 0
		delta_y = paths[i][1]-paths[i-1][1] if i!=0 else 0 
		delta_a = paths[i][-1]-paths[i-1][-1] if i!=0 else 0
		if (delta_x!=0 or delta_y!=0):
			changeFlag = True
		if delta_a !=0 and changeFlag==True:
			fea[-1] = 1

	# change position and then orientation
	changeFlag = False
	fea += [0]
	for i in range(len(paths)):
		delta_x = paths[i][0]-paths[i-1][0] if i!=0 else 0
		delta_y = paths[i][1]-paths[i-1][1] if i!=0 else 0 
		delta_a = paths[i][-1]-paths[i-1][-1] if i!=0 else 0
		if delta_a!=0:
			changeFlag = True
		if (delta_x!=0 or delta_y!=0) and changeFlag==True:
			fea[-1] = 1

		
	#the final place contains an object 
	[x,y,a] = paths[-1]	
	if (x,y) in env.nodes and env.nodes[(x,y)]!="":
		fea += [1]	
	else:
		fea += [0]

	#pass an object while walking
	fea += [0]
	for i in range(len(paths)):
		[x,y,a] =  paths[i]	
		if (x,y) in env.nodes and env.nodes[(x,y)]!="": 
			fea[-1] = 1
			break

	#the final place is a dead-end
	fea += [1]
	[x,y,a] = paths[-1]
	if (x,y) in env.edges: 
		if len(env.edges[x,y])<=1:
			fea[-1]=1
	
	#the final pose is the goal pose
	# ??? 
	fea += [0]

	#it is the first action to take
	if paths[0][-1] == -1:
		fea += [1]
	else:
		fea += [0]

	#final pose faces a new floor/wall color
	[x,y,a] = paths[-1]
	[x_p,y_p,a_p] = [x,y,a]
	trace = 1
	while trace < len(paths):
		[x_p,y_p,a_p] = paths[-1-trace]
		trace += 1
		if (x_p!=x or y_p!=y):
			break
	
	fea += [0, 0]
	if (x,y) in env.edges and (x_p!=x or y_p!=y):
		wall_next, wall_this, floor_next, floor_this = "", "", "", ""
		if a == 0:
			if (x,y-1) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x,y-1)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]
		elif a == 90:
			if (x-1,y) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x-1,y)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]
		elif a == 180:
			if (x,y+1) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x,y+1)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]
		elif a == 270:
			if (x+1,y) in env.edges[(x,y)]:
				[wall_next, floor_next] = env.edges[(x,y)][(x+1,y)]
				[wall_this, floor_this] = env.edges[(x_p, y_p)][(x,y)]		

		if wall_next != "" and wall_this!=wall_next:
			fea[-2]=1
		if floor_next != "" and floor_this != floor_next:
			fea[-1]=1
 
	#an object is visible from the final pose 
	[x,y,a] = paths[-1]
	fea += [0]
	if a == 0:
		while True:
			if (x,y-1) in env.nodes:
				if env.nodes[(x,y-1)] != "":
					fea[-1] = 1
			else:
				break
			y -= 1

	elif a == 90:
		while True:
			if (x-1,y) in env.nodes:
				if env.nodes[(x-1,y)] != "":
					fea[-1] = 1
			else:
				break
			x -= 1

			
	elif a == 180:
		while True:
			if (x,y+1) in env.nodes:
				if env.nodes[(x,y+1)] != "":
					fea[-1] = 1
			else:
				break
			y += 1


	elif a == 270:
		while True:
			if (x+1,y) in env.nodes:
				if env.nodes[(x+1,y)] != "":
					fea[-1] = 1
			else:
				break
			x += 1


	#there is an object at the turn 
	fea += [0]
	for i in range(len(paths)):
		if i !=0:
			[x,y,a] = paths[i]
			delta_a = a-paths[i-1][-1]
			if delta_a != 0 and (x,y) in env.nodes and env.nodes[(x,y)] != "":
				fea[-1]=1
				break

	
	#the place where to turn at is a dead-end
	fea += [0]
	for i in range(len(paths)):
		if fea[-1] == 1:
			break

		if i !=0:
			[x,y,a] = paths[i]
			delta_a = a-paths[i-1][-1]
			if delta_a != 0:
				if a == 0:
					while True:
						if (x,y-1) in env.nodes:
							length = len(env.edges[(x,y-1)])
							if length ==1:
								fea[-1] = 1
								break
							elif length >2:
								break
						else:
							break
						y -= 1
				elif a == 90:
					while True:
						if (x-1,y) in env.nodes:
							length = len(env.edges[(x-1,y)])
							if length ==1:
								fea[-1] = 1
								break
							elif length >2:
								break
						else:
							break
						x-=1
				elif a == 180:
					while True:
						if (x,y+1) in env.nodes:
							length = len(env.edges[(x,y+1)])
							if length ==1:
								fea[-1] = 1
								break
							elif length >2:
								break
						else:
							break
						y+=1
				elif a == 270: 				
					while True:
						if (x+1,y) in env.nodes:
							length = len(env.edges[(x+1,y)])
							if length ==1:
								fea[-1] = 1
								break
							elif length >2:
								break
						else:
							break
						x+=1

	return fea


def getFeatures(paths, i, env):
	global wallMap 
	global floorMap 
	global itemMap

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

tree = ET.parse(opts.input+"/SingleSentences-sail.xml")
root = tree.getroot()

commands_map = {}
sail_commands_map = {}
for examples in root:
	filename = examples.attrib["file"].lower()
	ID = examples.attrib['id'].lower()
	mapid = getMapID(examples.attrib["file"].lower())
	for example in examples:
		if example.tag == "nl":
			commands_map[filename+ID] = example.text.strip().lower().replace("\n"," ").strip()
		elif example.tag == "mrl":
			sail_commands_map[filename+ID] = example.text.strip().lower().replace("\n"," ").strip()

tree = ET.parse(opts.input+"/SingleSentences-marco.xml")
root = tree.getroot()

marco_commands_map = {}
for examples in root:
	filename = examples.attrib["file"].lower()
	ID = examples.attrib['id'].lower()
	mapid = getMapID(examples.attrib["file"].lower())
	for example in examples:
		if example.tag == "mrl":
			marco_commands_map[filename+ID] = example.text.strip().lower().replace("\n"," ").strip()

# read demonstrations and commands
tree = ET.parse(opts.demo_input)
root = tree.getroot()

paths_map = {}
features_map = {}
cas_features_map = {}
cas_subfeatures_map = {}

offset_map = {"l": 0, "jelly":100, "grid": 200}

filename = ""
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
			cas_features_map[filename+ID] = getPaperFeatures(path, envs[mapid])
			# path segment/ a state is a grid 
			cas_subfeatures_map[filename+ID] = []

			subpaths, subpath = [], []
			prev_x, prev_y, prev_a = -1,-1,-1
			for i in range(len(path)):
				[x,y,a] = path[i]
				if ( x!=prev_x or y!=prev_y ) and prev_x != -1:
					subpaths.append(subpath)
					subpath = []
				subpath.append(path[i])
				prev_x, prev_y, prev_a = x, y, a				
			if len(subpath)!=0:
				subpaths.append(subpath)				

			for i in range(len(subpaths)):
				[x,y,a] = subpaths[i][0]
				cas_subfeatures_map[filename+ID].append( [(x+offset_map[mapid],y+offset_map[mapid]), getStateFeatures(subpaths, i, envs[mapid])] )
				
		elif example.tag == "instruction":
			filename = example.attrib['filename'].lower()
	

features = []
paths = []
commands = []
sail_commands = []
cas_features = []
cas_subfeatures = []
marco_features = []
c = 0

memory = {}
for key in sail_commands_map:
	if key not in features_map:
		c += 1
		continue
	
	if opts.clear: 
		memory_key = commands_map[key] +";"+ sail_commands_map[key]
		if memory_key in memory:
			continue
		memory[memory_key] = True

	cas_features.append(cas_features_map[key])
	cas_subfeatures.append(cas_subfeatures_map[key])
	features.append(features_map[key])
	paths.append(paths_map[key])
	commands.append(commands_map[key])
	sail_commands.append(sail_commands_map[key])
	marco_features.append(getCASFeatures(marco_commands_map[key]))


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


map_features = []
for i in range(len(sail_process_commands)):
	fea = [0 for x in range(nDims)]
	for s in sail_process_commands[i]:
		fea[sail_map[s[0]][s[1]][s[2]]] = 1
	map_features.append(fea)


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
	savePickle(index_file, indexes)


def dumpPickle(path, data):
	output = open(path, 'wb')
	cPickle.dump(data, output)
	output.close()


for i in range(len(ratios)):
	start = int(len(features)*sum(ratios[:i]))
	end = int(len(features)*sum(ratios[:i+1]))

	print start, end

	paths_set = [paths[indexes[x]] for x in range(start, end)]
	sail_set = [sail_process_commands[indexes[x]] for x in range(start, end)]
	features_set = [features[indexes[x]] for x in range(start, end)]
	map_features_set = [map_features[indexes[x]] for x in range(start, end)]
	cas_features_set = [cas_features[indexes[x]] for x in range(start, end)]
	cas_subfeatures_set = [cas_subfeatures[indexes[x]] for x in range(start, end)]
	marco_features_set = [marco_features[indexes[x]] for x in range(start, end)]

	f = open(opts.target+"."+sets[i]+".txt","w")
	for j in range(start,end):
		f.write(commands[indexes[j]]+"\n")
	f.close()

	f = open(opts.source+"."+sets[i]+".txt","w")
	for j in range(start,end):
		f.write(sail_commands[indexes[j]]+"\n")
	f.close()

	dumpPickle(opts.source+".marcoFeature."+sets[i]+".pkl", marco_features_set)
	dumpPickle(opts.source+".path."+sets[i]+".pkl", paths_set)
	dumpPickle(opts.source+".sail."+sets[i]+".pkl", sail_set)
	dumpPickle(opts.source+".feature."+sets[i]+".pkl", features_set)
	dumpPickle(opts.source+".outFeature."+sets[i]+".pkl", map_features_set)
	dumpPickle(opts.source+".CASFeature."+sets[i]+".pkl", cas_features_set)
	dumpPickle(opts.source+".stateFeature."+sets[i]+".pkl", cas_subfeatures_set)
