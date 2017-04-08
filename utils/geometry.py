import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance
import copy

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
	vs += [[0,i,1./(2**0.5)] for i in [+1,-1]]

	# make an edge from each v to each other v
	edges = [(i,0) for i in range(1,4)] + [(1,2),(2,3),(3,1)]

	return vs, edges

def gen_3d_cube(noise=False, offset=None):
	vals = [-.5,.5]
	vs = [(e1,e2,e3) for e1 in vals for e2 in vals for e3 in vals]
	edges = [(1,2),(1,3),(4,3),(2,4),(5,6),(6,8),(7,8),(5,7)]
	edges += [(3,7),(4,8),(2,6),(1,5)]

	return vs,edges

def point_distance(p1,p2):
	return sum([(p1[i]-p2[i]) ** 2 for i in range(len(p1))]) ** 0.5

def are_intersecting(e1,e2,thresh):
	for v1 in e1[:2]:
		for v2 in e2[:2]:
			if point_distance(v1,v2) < thresh:
				return True
	return False 

def cluster_edges(unit_edges, thresh):
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

def edges_to_vertices(edges,thresh):
	''' given a list of edges in form :
		[(x1,y1,...),(x2,y2,...)]
		return all unique points that are within thresh distance from each 
		other

		note: man this is a bad approach to this problem... but fuckit
	'''
	vs = []
	for edge in edges:
		for v1 in edge[:2]:
			isnew = True
			for v2 in vs:
				if point_distance(v1,v2) < thresh:
					isnew = False
					break
			if isnew: vs.append(v1)
	return vs	

def cluster_to_mesh(cluster,thresh):
	''' given a cluster, 
		return a starting 3d mesh to project to 
		and a potentially new cluster of edges to project to 
		thresh: the tolerance for edges to count as intersecting
	'''
	# classify cluster
	# i dont know, do something better than this...
	p2ds = edges_to_vertices(cluster,thresh)
	if len(cluster) == 6:
		# its a tetrahedron! hopefully
		return [p2ds,cluster]+list(gen_3d_tetrahedron())
	if len(cluster) == 12:
		# it's maybe a cube!
		return [p2ds,cluster]+list(gen_3d_cube())

def project_3d_to_2d(C,p3ds):	
	points = []
	for p3d in p3ds:
		padded3d = np.append(p3d,1)
		padded2d = np.dot(C,padded3d)
		p2d = (padded2d/padded2d[-1])[:-1]
		points += [p2d]

	return np.array(points)

def write_element(x_struct,idxs,new_e):
	e = x_struct
	for i in idxs[:-1]:
		e = e[i]
	e[idxs[-1]] = new_e 

def get_element(x_struct,idxs):
	e = x_struct
	for i in idxs:
		e = e[i]
	return e

def detuple(lst):
	''' replace all tuples with lists recursively '''
	list_types = [list,type(np.array([1])), tuple]
	if type(lst) not in list_types:
		return lst
	else:
		return [detuple(e) for e in lst]

def fill_struct(x_struct,x0):
	''' fill the values of 1d array x0 in x_struct '''
	idxs = [0]  # an index stack pointing to current locatoin in struct
	x0_i = 0
	list_types = [list,type(np.array([1])), tuple]
	while idxs and idxs[-1] < len(get_element(x_struct,idxs[:-1])):
		e = get_element(x_struct, idxs)
		if type(e) not in list_types:
			# write element
			write_element(x_struct,idxs,x0[x0_i])
			x0_i += 1
		
			# update idxs
			# move up as much as necessary
			while idxs and idxs[-1]+1 == len(get_element(x_struct,idxs[:-1])):
				idxs.pop()
			if idxs:
				idxs[-1] += 1
		
		# go deeper
		while type(e) in list_types:
			idxs += [0]
			e = get_element(x_struct, idxs)

	return x_struct

def flatten(lst):
	list_types = [list,type(np.array([1])), tuple]
	if all([type(e) not in list_types for e in lst]):
		return lst
	else:
		new_lst = []
		for e in lst:
			if type(e) in list_types:
				new_lst += list(flatten(e))
			else: new_lst += [e]
		return new_lst

def geometry_loss(x, x_struct):
	# fill x into struct

	x_struct = fill_struct(x_struct, x)
	
	C, points_2d, points_3d, edges_2d, edges_3d = x_struct
	C = np.array(C)

	flat_points_2d = np.array(flatten(points_2d)).reshape(-1,2)
	flat_points_3d = np.array(flatten(points_3d)).reshape(-1,3)

	# the components of the loss function:

	# minimize distance between projected vertices and og vertices
	projected = project_3d_to_2d(C, flat_points_3d)
	# there's maybe prolly a way to do this faster...
	sum_dist = 0
	for i in range(len(projected)):
		sum_dist += distance.cosine(projected[i]+1e-10,flat_points_2d[i]+1e-10)
	
	# keep all edge lengths (of the 3d shapes) close to 1
	# TODO

	# keep avg distance of vertices close to origin
	# TODO
	# return 1. / (1+math.e**(-sum_dist))
	return sum_dist / len(projected)

def clusters_to_geometries(clusters,thresh):
	'''
	cluster are lists of 2d lines -> geometries are 3D points edges
	'''
	geometries = []	
	
	# seed a random camera matrix
	C = np.array([	[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0]])

	#C = np.random.random((3,4))

	# gen seed 3d models  	
	points_2d = []
	points_3d = []
	edges_2d = []
	edges_3d = []
	for cluster in clusters:
		# do i need to know the edges of the og 2d shape?
		p2ds, e2ds, p3ds, e3ds = cluster_to_mesh(cluster,thresh)
		points_2d += [p2ds]
		points_3d += [p3ds]
		edges_2d += [e2ds]
		edges_3d += [e3ds]
	
	x0_struct = [C, points_2d, points_3d, edges_2d, edges_3d]
	x_struct = detuple(copy.deepcopy(x0_struct))
	x0 = flatten(x0_struct)

	res = minimize(geometry_loss, x0, x_struct, 'Nelder-Mead')
	
	x_struct = fill_struct(x_struct, res.x)

	return x_struct


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

	clusters = cluster_edges(unit_edges,thresh) 

	# TODO
	# Translate to 3D
	geometries = clusters_to_geometries(clusters,thresh)


	return geometries

if __name__=='__main__':
	edges_1 = gen_tetrahedron_edges()
	edges = [[e[0],e[1],[e[2],e[2]]] for e in edges_1]
	mx = float(max(np.array(edges).max(),abs(np.array(edges).min())))
	thresh = 1./mx * 0.0001
	clusters = cluster_edges(edges,thresh)
	res = cluster_to_mesh(clusters[0],thresh)
	res = [[e] for e in res]
	p2ds, e2ds, p3ds, e3ds = res
	res = [p2ds, p3ds, e2ds, e3ds]
	C = np.array([  [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])

	#C = np.random.random((3,4))
	
	x0_struct = [C] + res
	x_struct = detuple(copy.deepcopy(x0_struct))
	x0 = flatten(x0_struct)
	'''
	C, points_2d, points_3d, edges_2d, edges_3d = x_struct
	flat_points_2d = np.array(flatten(points_2d)).reshape(-1,2)
	flat_points_3d = np.array(flatten(points_3d)).reshape(-1,3)
	projected = project_3d_to_2d(C, flat_points_3d)
	'''

	geometries = construct_geometry(edges_1)
	[C, points_2d, points_3d, edges_2d, edges_3d] = geometries

