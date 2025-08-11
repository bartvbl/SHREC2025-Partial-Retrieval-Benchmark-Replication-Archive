import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import knn

def backproject(mapping, point_cloud, pixel_features, device='cpu'):
    """
    Back-project features to points in the point cloud using the given mapping.

    :param mapping: tensor with shape (CANVAS_HEIGHT, CANVAS_WIDTH)
    :param point_cloud: tensor, points in the point cloud, with shape (#points, 3)
    :param pixel_features: features extracted with a backbone model. Shape is
        (CANVAS_HEIGHT, CANVAS_WIDTH, feature_dimensionality)
    :param device: device on which to place tensors and perform operations.
        Default to `cpu`

    :return: a new tensor of shape (#points, feature_dimensionality) that associates
        to each point in the point cloud a feature vector. Feature vector is all 0
        if no feature is being associated to a point.
        It can be indexed as `point_cloud`, so the i-th feature vector is associated
        to the i-th point in `point_cloud`.
    """
    # Feature vector: (points, feature_dimension)
    # For example: for 10,000 points and embedding dimension of 768,
    # features will be a (10000, 768) tensor
    features = torch.zeros((len(point_cloud), pixel_features.shape[-1]), dtype=torch.float, device=device)
    # Get pixel coordinates of pixels on which the point cloud has been projected
    yx_coords_of_pcd = (mapping != -1).nonzero()

    # Map features to the points
    # Explanation: `mapping` is a (HEIGHT, WIDTH) map with the same dimensionality
    # of the render. Each entry is either `-1` (no point has been mapped to the corresponding pixel)
    # or the index of the point in `point_cloud` that was mapped/projected to the corresponding
    # pixel.
    # So: `mapping != -1` returns a boolean mask to tell if a pixel is "empty" or if something
    # was projected there. Then, `mapping[mapping != -1]` returns the indices of the points
    # in `point_cloud` that have been mapped to a point. They're returned from top-left to
    # bottom-right order. Since `features` has a "row" for each point in `point_cloud`,
    # it can be indexed via the same indices as `point_cloud`. Therefore, `features[mapping[mapping != -1]]`
    # accesses the features of all the points that have been rendered (are visible) in the rendering.
    # Lastly, the assignment simply assigns features to those points. Features come from an "image"
    # (where number of channels is arbitrary, depending on the backbone model).
    # Note that a point may be mapped to multiple pixels, especially if using a large enough `point_size`.
    # In this case, a point will be assigned just the "last" features: if `pixel_features` has
    # two distinct feature vectors (e.g. [1, 2] and [3, 4]) for the point (x, y), the point (x, y)
    # will be ultimately assigned features [3, 4]. While this may sound like a problem, it is actually
    # not in most practical applications: if a point is projected into multiple pixels, they are certainly
    # neighbouring pixels. Therefore, they most likely have very similar feature vectors: so overwriting
    # the features and just keeping the "last" that comes (usually the most bottom-right in the
    # `pixel_features` "image") is not an actual problem.
    features[mapping[mapping != -1]] = pixel_features[yx_coords_of_pcd[:, 0], yx_coords_of_pcd[:, 1]]

    return features


def backproject_on_existing_tensor(mapping, features, pixel_features):
    """
    Back-project features to points in the point cloud using the given mapping.

    :param mapping: tensor with shape (CANVAS_HEIGHT, CANVAS_WIDTH)
    :param features: feature vector to store features onto
    :param pixel_features: features extracted with a backbone model. Shape is
        (CANVAS_HEIGHT, CANVAS_WIDTH, feature_dimensionality)

    :return: indices of points that received a feature vector
    """

    # Get pixel coordinates of pixels on which the point cloud has been projected
    mask = (mapping != -1)
    yx_coords_of_pcd = mask.nonzero()
    points = mapping[mask]

    # Map features to the points
    features[points] += pixel_features[yx_coords_of_pcd[:, 0], yx_coords_of_pcd[:, 1]]

    return points


def interpolate_point_cloud(pcd, features, points_with_missing_features=None, neighbors=10, copy_features=False, zero_nan=True):
    """
    Interpolate features associate to each point for points that are missing them.
    A point misses a feature if all the feature vector associated to it is made up
    of NaN values.
    Features are therefore interpolated using neighboring points.

    Note that if all the closest neighbors of a point are missing features as well,
    then the feature vector will still be composed only of NaN values. The `zero_nan` parameter allows to control
    this behavior: setting it to True (which is the default value) will substitute all NaN values with 0.

    :param pcd: the point cloud tensor, with shape (#points, 3)
    :param features: the features. Each element is a tensor representing the features associated to the
        point with the same index in `pcd`
    :param points_with_missing_features: indices of points which are missing features. If None, it will be
        automatically determined by finding all the feature vectors which have all features (along feature
        dimension) set to 0.0. Default to None
    :param neighbors: how many neighbors to consider in the interpolation
    :param copy_features: whether to copy the feature tensor or modify directly the one being passed
    :param zero_nan: if a feature vector is still NaN after interpolation (because all its neighbors are NaN, too),
        set it to 0 anyway. True: set NaN to 0 after interpolation. Defaults to True
    :return: the interpolated tensor
    """

    if copy_features:
        features = copy.deepcopy(features)

    if points_with_missing_features is None:
        points_with_missing_features = torch.all(features == 0, dim=-1).nonzero().view(-1)

    if len(points_with_missing_features) == 0:
        return features

    # k-NN between all the points
    neighbors += 1
    knn_on_cluster_assignment = knn(pcd, pcd, neighbors)

    # Get the features only for the points that will have to be used to interpolate feature
    # vectors, and then compute the average between all the neighbors for each point
    knn_on_cluster_assignment = knn_on_cluster_assignment[1].view(len(pcd), neighbors)
    neighbors_features = features[knn_on_cluster_assignment[points_with_missing_features].view(-1)].view(
        len(points_with_missing_features), neighbors, -1
    )
    # Setting to NaN makes it possible to compute the average only of those points which actually
    # have features. If this was left to 0, they would influence the mean, while this makes
    # it possible not to take them into account
    neighbors_features[torch.all(neighbors_features == 0, dim=-1)] = float('nan')
    features[points_with_missing_features] = neighbors_features.nanmean(dim=1)

    # Adjust feature vectors for points whose neighbors are all NaN (no feature vector assigned to them)
    if zero_nan:
        features = torch.nan_to_num(features)

    return features


def interpolate_feature_map(features, width, height, mode='bicubic'):
    """
    Interpolate a patchy feature map to the specified size (width, height)

    :param features: tensor of shape (batch_size, #patches + 1 for [CLS], embedding_dimension)
    :param width: width of the output
    :param height: height of the output
    :param mode: interpolation method. Default to 'bicubic'
    :return: interpolated feature map
    """

    # if isinstance(features, dict) and 'last_hidden_state' in features.keys():
    #     features = features.last_hidden_state
    # features = features[:, 1:, :]

    # R: renders
    # L: length
    R, L, _ = features.shape
    W = H = np.sqrt(L).astype(int)

    with torch.no_grad():
        interpolated_features = F.interpolate(
            features.view(R, W, H, -1).permute(3, 0, 1, 2),
            size=(width, height),
            mode=mode,
            align_corners=False if mode not in ['nearest', 'area'] else None,
        )
    interpolated_features = interpolated_features.permute(1, 2, 3, 0)

    return interpolated_features

def aggregate_features(features, mappings, point_cloud, interpolate_missing=True, average=True, device='cpu'):
    """
    Aggregate features from different views for each point in the point cloud

    :param features: features from a backbone 2D model. Shape: (#views, width, height, embedding_dimensionality)
        #views is the same as len(mappings), as each feature map comes with its corresponding mapping
    :param mappings: tensor mapping from pixels to a point in `point_cloud`. Each "pixel"
        in `mappings` provides the index to a point in `point_cloud`, or -1 if it does not
        map to anything. Shape is (#views, width, height)
    :param point_cloud: tensor of shape (#points, 3) or (#points, 6)
    :param interpolate_missing: whether to interpolate missing features for points after aggregation.
        For more control, set this to False and call `interpolate_point_cloud` instead, passing all
        the desired parameters. Default to True
    :param average: whether to average the features collected for each point or not. Default to True
    :param device: where to perform the computation. Default to 'cpu'
    :return: the aggregated features, one feature vector per point in `point_cloud`
    """

    # Instead of stacking point-associated features and then using torch.nan_mean() on them,
    # this code aggregates them progressively to avoid consuming too much VRAM, which causes
    # the GPU to run out of memory with very large point clouds (>110_000 points).
    # This code can handle efficiently even larger points clouds (tested up to >230_000 points).
    feature_pcd_aggregated = torch.zeros((len(point_cloud), features.shape[-1]), device=device, dtype=torch.double)
    count = torch.zeros(len(feature_pcd_aggregated), device=device)

    for features_from_view, mapping in zip(features, mappings):
        feature_pcd = backproject(mapping, point_cloud, features_from_view, device=device)
        nan_mask = ~torch.all(feature_pcd == 0.0, dim=-1)
        feature_pcd_aggregated[nan_mask] += feature_pcd[nan_mask]
        count[nan_mask] += 1

    # Avoid division by zero
    count[count == 0] = 1

    # Compute the mean
    if average:
        feature_pcd_aggregated /= count.unsqueeze(-1)

    # Interpolate features
    if interpolate_missing:
        feature_pcd_aggregated = interpolate_point_cloud(point_cloud[:, :3], feature_pcd_aggregated, neighbors=20)

    return feature_pcd_aggregated
