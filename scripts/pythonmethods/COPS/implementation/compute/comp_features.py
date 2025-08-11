import torch
# features utils
from implementation.utils import get_min_camera_distance, improved_view_points, sample_view_points
from implementation.utils import DINOWrapper
#types
from pytorch3d.renderer import look_at_view_transform
#cops
from implementation.cops import FeatureExtractor

def compute_features_cops(verts: torch.Tensor, model: DINOWrapper, device: str) -> torch.Tensor:
    verts = verts.to(device)
    model = model.to(device)
    #############################################################
    # variables para calculo de features
    res = 224 
    fov = 60
    partitions = 1
    #############################################################
    dist_farthest = verts.norm(dim=-1).max().item()
    render_dist = get_min_camera_distance(dist_farthest, fov) * 1.2 # para tener un margen
    views = sample_view_points(render_dist, partitions)
    rotations, translations = look_at_view_transform(eye=views, device=device)
    # calcula sus caracteristicas
#    print("Calculando features cops")
    feature_extractor = FeatureExtractor(model, rotations, translations, device, canvas_height=res, canvas_width=res)
    features = feature_extractor(verts, None)
    return features

