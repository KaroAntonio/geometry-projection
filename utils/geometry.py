import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def add_custom(vs,noise,offset):
    if noise:
        dx = random.random() * 0.1
        dy = random.random() * 0.1
        vs =  [[x + dx,y + dy] for x,y in vs]

    if offset:
        vs = [[x + offset[0],y + offset[1]] for x,y in vs]
    return vs

def gen_cube_edges(noise=False,offset=None):
    ''' generate n cubes '''

    vs = [(0,0),(-2,1),(-2,-1),(0,-2),(2,-1),(2,1),(0,2)]

    vs = add_custom(vs, noise, offset)

    edge_idx = [(0,i,0) for i in [1,3,5]]+[(0,i,-1) for i in [2,4,6]]
    edge_idx += [(1,2,0),(2,3,0),(3,4,0),(4,5,0),(5,6,0),(6,1,0)]

    return [[vs[e[0]],vs[e[1]],e[2]] for e in edge_idx]

def gen_tetrahedron_edges(noise=False,offset=None):

    vs = [(0,0),(0,2),(1.5,-1),(-1.5,-1)]

    vs = add_custom(vs, noise, offset)

    edge_idx = [(0,i,-1) for i in [1,2,3]] + [(1,2,0),(2,3,0),(3,1,0)]

    return [[vs[e[0]],vs[e[1]],e[2]] for e in edge_idx]

def gen_3d_tetrahedron(noise=False,offset=None):
	vs = [[i,0,-1./(2**0.5)] for i in [+1,-1]]
	vs += [[0,i,-1./(2**0.5)] for i in [+1,-1]]

	# make an edge from each v to each other v
	edges = []
	for i in range(len(vs)):
		for j in range(len(vs)):
			edges += [[i,j]]

	return vs, edges

def gen_3d_cube(noise=False, offset=None):
	vals = [-.5,.5]
	vs = [(e1,e2,e3) for e1 in vals for e2 in vals for e3 in vals]
	

def distance(p1,p2):
	return sum([(p1[i]-p2[i]) ** 2 for i in range(len(p1))]) ** 0.5

def are_intersecting(e1,e2,thresh):
	for v1 in e1[:2]:
		for v2 in e2[:2]:
			if distance(v1,v2) < thresh:
				return True
	return False 

def find_meshes(unit_edges, thresh):
	# unconnected edges are inserted as single edge meshes
	unit_edges = unit_edges[:]
	meshes = []
	while unit_edges:
		# START BUILDING MESH 
		# (traversing graph)
		# starting edge
		mesh = []
		unexplored = [unit_edges.pop()]
		while unexplored:
			e1 = unexplored.pop()
			i = 0
			while unit_edges and i < len(unit_edges):
				e2 = unit_edges[i]
				if are_intersecting(e1,e2,thresh):
					unit_edges.pop(i)
					unexplored += [e2]
				else:
					i += 1
			mesh += [e1]
		meshes += [mesh]
	return meshes

def show_2d_geom(edges):
	xs = [ e[0] for e in edges]
	ys = [ e[1] for e in edges]
	for e in edges:
		xs = [e[0][0],e[1][0]]
		ys = [e[0][1],e[1][1]]
		c = 'r' if e[2][0] else 'k'
		plt.plot(xs,ys, color=c, linestyle='-', linewidth=2)
	plt.show()

def num_occluded(edges):
	return sum([1 for e in edges if e[2][0] < 0])

def mesh_centre(edges):
	''' return the avged centre for the mesh '''
	x_sum = [e[0][0]+e[1][0] for e in edges]	

	return x_sum / (len(edges)*2.) , s_sum / (len(edges)*2.)

def points(cluster,thresh):
	# return all vertices in the cluster (within a certain thresh
	pass


def cluster_to_mesh(cluster,thresh):
	''' given a cluster, 
		return a starting 3d mesh to project to 
		and a potentially new cluster of edges to project to 
		thresh: the tolerance for edges to count as intersecting
	'''
	# classify cluster
	# i dont know, do something better than this...
	if len(cluster) == 6:
		# its a tetrahedron! hopefully
		return gen_tetrahedron

def geometry_loss(x):
	C = x[0]

	# the components of the loss function:
	# distance between projected edges (points) and 

def cluster_to_geometries(clusters):
	'''
	cluster are lists of 2d lines -> geometries are lists of 3d edges
	'''
	geometries = []	
	
	# seed a random camera matrix
	C = np.array([	[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0]])

	# gen seed 3d models  	
	points_2d = []
	points_3d = []
	for cluster in clusters:
		p2ds, p3ds = cluster_to_mesh(cluster)
		points_2d += [p2ds]
		points_3d += [p3ds]

	x0 = [C, seed_meshes, seed_clusters]
	shapes = [len(e) for e in x0] 

	res = minimize(geometry_loss, x0, shapes, 'BFGS')

	

	'''
	for mesh in meshes:
		# a really naive approach
		geom = []
		for edges in mesh:
			if len(mesh) == 6:
				# tetrahedron somehow decide to build a tetrahedron
				# super janky
				n_occ = num_occluded(edges)
				if n_occ == 0:
					# then there are no occluded edges and the centre is out of plane	
	'''

def construct_geometry(edges, thresh = None):
	'''
	edges: [(v1,v2,z1),(v3,v4,z2),...]  tuples of vertices , 
									the vertices may or may not be unique
									z for each edge is it's z-index
										for obstruction

	For the given vertices and edges, 
		determine the shape that each set of edges belongs too,
		fill in missing edges and vertices
	 	
	return geometries: [(vertices,edges),...]
	'''
	# Normalize 
	# convert all edges to unit, more or less
	if type(edges[0][2]) not in [list, tuple]:
		# transform edge format (v1, v2, z) -> (v1, v2, [z,z])
		edges = [[e[0],e[1],[e[2],e[2]]] for e in edges]

	mx = float(max(np.array(edges).max(),abs(np.array(edges).min())))
	#unit_edges = [[[v[0]/mx,v[1]/mx] for v in e]+[e[2]] for e in edges]
	unit_edges = edges
	
	show_2d_geom(unit_edges)

	if not thresh:
		thresh = 1./mx * 0.0001

	meshes = find_meshes(unit_edges,thresh) 

	# TODO
	# Translate to 3D

	return meshes

def geometries_to_obj(geoms):
	pass


