import numpy as np
import sys
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from implementation.utils import DINOWrapper
from implementation.compute import compute_features_cops
from pytorch3d.ops import sample_farthest_points
from queue import Queue
import threading

device = None
_model_pool = None
_num_models = 0

def init_model_pool(num_models=1, device='cuda:0'):
	global _model_pool
	global _num_models
	_num_models = num_models
	_model_pool = Queue()
	for _ in range(num_models):
		_model_pool.put(DINOWrapper(device=device, small=True, reg=True))

def get_model():
    return _model_pool.get()

def release_model(model):
    _model_pool.put(model)

def get_used_model_count():
    return _num_models - _model_pool.qsize()

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

def initMethod(config):
	print('COPS method active!')
	global device
	gpu_device = "cuda:0"
	device = torch.device(gpu_device)
	init_model_pool(num_models=6, device=device)

def destroyMethod():
	print('COPS method destroyed')

def getMethodMetadata():
	return {}

def computeMeshDescriptors(mesh, descriptorOrigins, config, supportRadii, randomSeed):
	return np.zeros((descriptorOrigins.shape[0], 32), dtype=float)

def computePointCloudDescriptors(pointCloud, descriptorOrigins, config, supportRadii, randomSeed):
	# sample 10000 points using torch
	set_seed(randomSeed)
	points = torch.from_numpy(pointCloud.vertices).to(device)
	points = points.unsqueeze(0)
	point_cloud, _ = sample_farthest_points(points, K=10000) #use less?
	point_cloud = point_cloud.squeeze(0)
	# add descripotorOrigins to point_cloud
	vertices_torch = torch.from_numpy(descriptorOrigins[:, 0, :]).to(device) # shape (n, 3)
	point_cloud = torch.cat([point_cloud, vertices_torch], dim=0)  # shape (N+n, 3)
	# computes the features
	model = get_model()
	real_return = None
	#print(randomSeed, "MODELS IN USE:", get_used_model_count())
	try:
		feature_pcd_aggregated = compute_features_cops(point_cloud, model, device)
		real_return = feature_pcd_aggregated[-descriptorOrigins.shape[0]:, :].cpu().numpy()
	except:
		real_return = np.zeros((descriptorOrigins.shape[0], 384), dtype=float)
	finally:
		release_model(model)
	#print(randomSeed, "COPS features computed")
	return real_return