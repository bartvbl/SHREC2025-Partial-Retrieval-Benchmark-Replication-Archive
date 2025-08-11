from .views import get_min_camera_distance, sample_view_points, improved_view_points
from .model_wrappers import DINOWrapper

models = {
    "dino": DINOWrapper,
}