import numpy as np

from utils.ap_calculator import parse_predictions


BOX_EDGES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for --display_bounding_boxes. Install it with `pip install open3d`."
        ) from exc
    return o3d


def visualize_predictions_with_open3d(
    point_clouds,
    predicted_box_corners,
    sem_cls_probs,
    objectness_probs,
    ap_config_dict,
    sample_idx=0,
    window_name="3DETR Predictions",
):
    o3d = _import_open3d()

    batch_pred_map_cls = parse_predictions(
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        point_clouds,
        ap_config_dict,
    )

    point_cloud = point_clouds[sample_idx, :, :3].detach().cpu().numpy()
    point_cloud = point_cloud.copy()
    point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    point_cloud[:, 1] *= -1
    sample_predictions = batch_pred_map_cls[sample_idx]

    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    geometries.append(pcd)

    for _, corners, _ in sample_predictions:
        corners = np.asarray(corners)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(BOX_EDGES)
        colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(BOX_EDGES), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)
    return len(sample_predictions)
