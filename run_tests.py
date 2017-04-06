from utils.geometry import *
import numpy as np
import random

if __name__ == '__main__':
	edges_1 = gen_cube_edges()
	edges_2 = gen_cube_edges(offset=(1,0))
	edges_3 = gen_tetrahedron_edges(offset=(1,1))
	edges = edges_3 + edges_1 + edges_2
	meshes = construct_geometry(edges)
