import pickle
import traceback

import numpy as np
from gedi import GeDi
import torch
import math
from queue import Queue
import threading
import os.path

_enable_debug_dumps = True
_model_pool = None
_num_models = 0
_num_GEDI_failures = 0
_num_GEDI_descriptors = 0
_num_NAN_failures = 0
_num_NAN_values = 0
_num_NAN_descriptors = 0

def init_model_pool(num_models=1):
	global _model_pool
	global _num_models
	_num_models = num_models
	_model_pool = Queue()
	config = {'dim': 32,                                            # descriptor output dimension
              'samples_per_batch': 500,                             # batches to process the data on GPU
              'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
            'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
            'r_lrf': .5,                                          # LRF radius
            'fchkpt_gedi_net': '../scripts/pythonmethods/GEDI/chkpt.tar'}   # path to checkpoint

	for _ in range(num_models):
		_model_pool.put(GeDi(config))

def get_model():
	return _model_pool.get()

def release_model(model):
	return _model_pool.put(model)

def get_used_model_count():
	return _num_models - _model_pool.qsize()

def initMethod(config):
	print('GEDI method active!')
	init_model_pool(num_models=6)

def destroyMethod():
	print('GEDI method destroyed')

def getMethodMetadata():
	return {}

def set_seed(seed):
	seed = seed % (2**32)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# Enforce deterministic behavior
	torch.use_deterministic_algorithms(True)
	# Optional: for deterministic cudnn backend
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def computeMeshDescriptors(mesh, descriptorOrigins, config, supportRadii, randomSeed):
	#print("Processing mesh", mesh.vertices.shape, mesh.normals.shape, descriptorOrigins.shape)
	vertexCount = mesh.vertices.shape[0]
	numberOfDescriptorsToCompute = descriptorOrigins.shape[0]
	descriptorOriginVertex = descriptorOrigins[0][0]
	descriptorOriginNormal = descriptorOrigins[0][1]

	return np.zeros((descriptorOrigins.shape[0], 32), dtype=float)

def computePointCloudDescriptors(pointCloud, descriptorOrigins, config, supportRadii, randomSeed):
	global _num_GEDI_failures
	global _num_GEDI_descriptors
	global _num_NAN_failures
	global _num_NAN_values
	global _num_NAN_descriptors
	global _enable_debug_dumps

	set_seed(randomSeed)
	input_point_cloud = np.asarray(pointCloud.vertices)
	points_to_describe = np.asarray(descriptorOrigins[:, 0])
	#print("Processing cloud", input_point_cloud.shape, points_to_describe.shape)

	flag = 0
	#Repeat once points_to_describe if it only contains one point
	if points_to_describe.shape[0] == 1:
		flag = 1
		points_to_describe = np.repeat(points_to_describe, 2, axis=0)


	gedi = get_model()

	pts0 = torch.tensor(points_to_describe).float()
	_pcd0 = torch.tensor(np.asarray(input_point_cloud)).float()
	try:
		desc = gedi.compute(pts=pts0, pcd=_pcd0)
	except Exception as e:
		# Seems to only be needed when region is empty. Using this as a fallback.
		desc = np.zeros((len(pts0), gedi.dim))
		if _enable_debug_dumps:
			output_dir = os.path.join('failures', 'exception_thrown', 'id_{}_seed_{}'.format(_num_GEDI_failures, randomSeed))
			os.makedirs(output_dir, exist_ok=True)
			input_point_cloud = np.asarray(pointCloud.vertices)
			np.save(os.path.join(output_dir, 'input_cloud.npy'), input_point_cloud)
			np.save(os.path.join(output_dir, 'vertices.npy'), descriptorOrigins)
			with open(os.path.join(output_dir, 'exception.txt'), 'w') as f:
				traceback.print_exc(file=f)
		#print('execution of GEDI failed. Error:', e)
		#print('Used random seed:', randomSeed)
		#print('Used support radii:', supportRadii)
		#print('Used descriptor origins:', descriptorOrigins)
		_num_GEDI_failures += 1
		_num_GEDI_descriptors += len(supportRadii)
		print()
		print('A total of {} failures have been detected during this run, lost descriptor count: {}'.format(_num_GEDI_failures, _num_GEDI_descriptors))

	finally:
		release_model(gedi)

	nancount = np.count_nonzero(np.isnan(desc))
	if nancount > 0:
		# Happens when mesh is completely flat
		desc = np.zeros((len(pts0), gedi.dim))

		if _enable_debug_dumps:
			output_dir = os.path.join('failures', 'nans_detected', 'id_{}_seed_{}'.format(_num_NAN_failures, randomSeed))
			os.makedirs(output_dir, exist_ok=True)
			input_point_cloud = np.asarray(pointCloud.vertices)
			np.save(os.path.join(output_dir, 'input_cloud.npy'), input_point_cloud)
			np.save(os.path.join(output_dir, 'vertices.npy'), descriptorOrigins)
		#print('Used random seed:', randomSeed)
		#print('Used support radii:', supportRadii)
		#print('Used descriptor origins:', descriptorOrigins)
		_num_NAN_failures += 1
		_num_NAN_values += nancount
		_num_NAN_descriptors += len(supportRadii)
		print()
		print('A total of {} NaNs were detected in the produced descriptors! Total {} values in {} failures, {} descriptors'.format(nancount, _num_NAN_values, _num_NAN_failures, _num_NAN_descriptors))

	if flag == 1:
		desc = desc[0:1, :]

	return desc


