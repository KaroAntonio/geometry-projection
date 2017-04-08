import numpy as np
import itertools

def adj_mat(edges):
	''' transform edges list to adj matrix '''

	s = np.array(edges).max()+1
	adj_mat = np.zeros((s,s))
	for v1,v2 in edges:
		adj_mat[v1][v2] = 1
		adj_mat[v2][v1] = 1

	return adj_mat

def neighbours(G, v0):
	return [v1 for v1 in range(len(G)) if G[v0][v1] and v1 != v0]

def is_valid_face(G,face):
	# TODO validate

	used_vs = {0:0}
	curr_v = face[0]
	while len(used_vs) < len(face):
		nbrs = [v for v in neighbours(G,curr_v) if v in face]
		if len(nbrs) != 2:
			return False
		# removed visited nbrs
		nbrs = [v for v in nbrs if v not in used_vs] 
		if not nbrs and len(used_vs) == len(face):
			return True
		curr_v = nbrs[0]
		used_vs[curr_v] = 0 

	if len(used_vs) == len(face): return True
	return False

def edges_to_faces(verts, edges, max_len=4):
	'''
	verts: [(x1,y1,z1),...]
	edges: [(v1,v2),(v3,v4)...]

	given vertices and edges, 
	return the list of triangular faces in form
	faces: [(v1, v2, v3) ... ]
	where necessary, add edges to ensure all faces are triangular
	'''
	G = adj_mat(edges)	

	# for all verts
	faces = []
	for n in range(3,max_len):
		unique_verts = [i for i in range(len(G))]
		for face in itertools.combinations(unique_verts, n):
			face = list(face)
			# check that the face is a hamcycle
			if is_valid_face(G,face):
				#face.reverse()
				faces += [face]

	# TODO 
	# for every face of len(face) > 3, split face into triangles

	return faces
			
def load_obj(fid):
	''' load from an obj file '''
	f = open(fid)
	objs = {}
	curr_obj = None
	curr_obj_id = None
	for line in f:
		if line.strip():
			token = line.strip().split()[0]
		else: 
			token = ''
			line = ''

		if token == 'g':
			# START NEW OBJ
			curr_obj_id = line.strip().split()[1].strip()
			# one list for each of vertices and faces
			objs[curr_obj_id] = [[],[]] 
			curr_obj = objs[curr_obj_id]
		if token == 'v':
			coords = [float(e) for e in line.strip().split()[1:]]
			curr_obj[0] += [coords]
		if token == 'f':
			v_idxs = [int(e) for e in line.strip().split()[1:]]
			curr_obj[1] += [v_idxs]
	f.close()
	return objs

def save_obj(objs,fid):
	f = open(fid,'w')
	
	for obj_id in objs:
		f.write('g {}\n\n'.format(obj_id))	
		obj = objs[obj_id]

		# VERTICES
		for v in obj[0]:
			f.write('v {} {} {}\n'.format(*v))

		f.write('\n')
		
		# FACES
		for face in obj[1]:
			f.write('f {} {} {}\n'.format(*face))

		f.write('\n')

	f.close()

if __name__ == '__main__':
	objs = load_obj('cube_basic.obj')
	save_obj(objs, 'saved_obj.obj')
	
	from geometry import *

	vs,edges = gen_3d_tetrahedron()
	G = adj_mat(edges)

	faces = edges_to_faces(vs,edges)

	objs = {}
	objs['tetra'] = [vs,faces]

	save_obj(objs, 'saved_obj.obj')


	

