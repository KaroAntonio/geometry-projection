from utils.geometry import *
from utils.obj import *
import numpy as np
import random

if __name__ == '__main__':
	edges_1 = gen_tetrahedron_edges()
	geometries = construct_geometry(edges_1)
	[C, points_2d, points_3d, edges_2d, edges_3d] = geometries

	faces = edges_to_faces(points_3d[0], edges_3d[0])

	objs = {}
	objs['tetra'] = [vs,faces]

	save_obj(objs, 'saved_obj.obj')

