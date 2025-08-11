import torch
import torch.nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    NormWeightedCompositor
)

class PointsRendererWithFragments(torch.nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.

    The points are rendered with varying alpha (weights) values depending on
    the distance of the pixel center to the true point in the xy plane. The purpose
    of this is to soften the hard decision boundary, for differentiability.
    See Section 3.2 of "SynSin: End-to-end View Synthesis from a Single Image"
    (https://arxiv.org/pdf/1912.08804.pdf) for more details.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments

def render_and_map_pytorch3d(
        point_cloud,
        rotations, translations,
        canvas_width=600, canvas_height=600,
        point_size=0.007,
        points_per_pixel=10,
        perspective=True,
        device='cpu'
    ):

    verts = point_cloud
    verts -= point_cloud.mean(dim=0)
    verts /= verts.norm(dim=-1).max()
    rgb = point_cloud / 255

    point_cloud_object = Pointclouds(points=[verts], features=[rgb])
    point_cloud_stacked = point_cloud_object.extend(len(rotations))

    # Prepare the cameras
    if perspective:
        cameras = FoVPerspectiveCameras(device=device, R=rotations, T=translations, znear=0.01)
    else:
        cameras = FoVOrthographicCameras(device=device, R=rotations, T=translations, znear=0.01)

    # Prepare the rasterizer
    raster_settings = PointsRasterizationSettings(
        image_size=(canvas_width, canvas_height),
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=0
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRendererWithFragments(
        rasterizer=rasterizer,
        compositor=NormWeightedCompositor(background_color=(1., 1., 1.))
    )

    # Get mappings and rendered images
    rendered_images, fragments = renderer(point_cloud_stacked)
    for idx in range(len(rotations)):
        fragments.idx[idx, fragments.idx[idx] != -1] -= (idx * len(verts))

    return rendered_images[..., :3], fragments.zbuf[..., 0], fragments.idx[..., 0]