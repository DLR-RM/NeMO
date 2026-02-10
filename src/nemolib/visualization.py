"""Functions to help visualizing data.
Parts of of this file are taken from https://github.com/naver/dust3r/blob/main/dust3r/viz.py"""

from typing import Callable, Literal, Optional, Sequence, Union

from PIL import Image
import io
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from einops import rearrange
import cv2

from .utils import confidence_to_imgs

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def draw_3d_box(image, pose_4x4, intrinsic, scale=0.1, hide_back_faces=True, color=None):
    # 8 cube corners in object coordinates
    s = scale / 2
    cube = np.float32([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ])

    # Cube edges (pairs of vertex indices)
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # back face
        (4,5),(5,6),(6,7),(7,4),   # front face
        (0,4),(1,5),(2,6),(3,7)    # side edges
    ]

    # Extract pose
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3].reshape(3, 1)

    # Project points
    # cv2.projectPoints needs rvec, tvec
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(cube, rvec, t, intrinsic, None)
    proj = proj.reshape(-1, 2).astype(int)

    # Compute depth of each vertex in camera frame
    cube_cam = (R @ cube.T + t).T  # Nx3
    depths = cube_cam[:, 2]       # z-values

    # --- Optionally remove edges whose faces are back-facing ---
    if hide_back_faces:
        visible_edges = []
        for i, j in edges:
            if depths[i] > 0 and depths[j] > 0:
                # keep only edges whose both endpoints face camera
                visible_edges.append((i, j))
        edges_to_draw = visible_edges
    else:
        edges_to_draw = edges

    # --- Sort edges from far → near for correct depth cue ---
    # Use mean depth of each edge
    edges_sorted = sorted(
        edges_to_draw,
        key=lambda e: (depths[e[0]] + depths[e[1]]) / 2,
        reverse=True  # far edges drawn first
    )

    # Draw edges with depth-based coloring
    z_min, z_max = np.min(depths), np.max(depths)
    z_range = z_max - z_min + 1e-6

    for i, j in edges_sorted:
        # depth normalized: far edges = darker, near edges = bright
        mean_z = (depths[i] + depths[j]) * 0.5
        alpha = 1.0 - (mean_z - z_min) / z_range  # near=1, far=0

        c = (
            int(255 * alpha),
            int(255 * alpha),
            int(255 * (0.4 + 0.6 * alpha))  # bluish depth shading
        ) if not color else color

        cv2.line(image, tuple(proj[i]), tuple(proj[j]), c, 2)

    return image



def draw_pose_axes(image, pose_4x4, intrinsic, axis_length=0.1):
    """
    Draws a 3D pose (X-Y-Z axes) on a 2D image.

    pose_4x4: 4×4 object pose in camera coordinates ( R | t ).
    intrinsic: 3×3 camera intrinsics.
    """

    # Extract rotation and translation
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3].reshape(3, 1)

    # Axis endpoints in 3D
    axes_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],   # X - red
        [0, axis_length, 0],   # Y - green
        [0, 0, axis_length]    # Z - blue
    ])

    # Project 3D axes to 2D
    points_2d, _ = cv2.projectPoints(axes_3d, cv2.Rodrigues(R)[0], t, intrinsic, None)
    points_2d = points_2d.reshape(-1, 2).astype(int)

    origin = tuple(points_2d[0])
    x_end = tuple(points_2d[1])
    y_end = tuple(points_2d[2])
    z_end = tuple(points_2d[3])

    # Draw arrows for axes
    cv2.arrowedLine(image, origin, x_end, (0,0,255), 2)  # X: red
    cv2.arrowedLine(image, origin, y_end, (0,255,0), 2)  # Y: green
    cv2.arrowedLine(image, origin, z_end, (255,0,0), 2)  # Z: blue

    return image


def to_numpy(x): return todevice(x, 'numpy')


def cat(a, b):
    return np.concatenate((a.reshape(-1, 3), b.reshape(-1, 3)))


OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]

def get_opencv_camera_frustum(R, t, scale=0.1, size=2, color="blue", image=None):
    """
    Generate a frustum using OpenCV camera convention and optionally overlay an image.

    Args:
        R: (3x3) rotation matrix (world_from_camera)
        t: (3,) camera position in world coordinates
        scale: frustum depth
        size: line thickness
        color: color of frustum lines
        image: optional (H, W, 3) image to project into frustum
    Returns:
        list of Plotly 3D traces
    """
    if isinstance(color, Sequence):
        color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
    # Frustum corners in camera space (scaled)
    frustum_corners_cam = (
        np.array(
            [
                [1, -1, 2],  # top-right
                [-1, -1, 2],  # top-left
                [-1, 1, 2],  # bottom-left
                [1, 1, 2],  # bottom-right
            ]
        ).T
        * scale
    )

    # Transform to world space: X_world = R @ X_cam + t
    corners_world = (R @ frustum_corners_cam) + t.reshape(3, 1)
    center_world = t.reshape(3, 1)

    traces = []

    # Lines from camera center to corners
    for i in range(4):
        traces.append(
            go.Scatter3d(
                x=[center_world[0, 0], corners_world[0, i]],
                y=[center_world[1, 0], corners_world[1, i]],
                z=[center_world[2, 0], corners_world[2, i]],
                mode="lines",
                line=dict(color=color, width=size),
                showlegend=False,
            )
        )

    # Lines between corners (edges of image plane)
    for i in range(4):
        j = (i + 1) % 4
        traces.append(
            go.Scatter3d(
                x=[corners_world[0, i], corners_world[0, j]],
                y=[corners_world[1, i], corners_world[1, j]],
                z=[corners_world[2, i], corners_world[2, j]],
                mode="lines",
                line=dict(color=color, width=size),
                showlegend=False,
            )
        )

    # Optional: draw the center as a dot
    traces.append(
        go.Scatter3d(
            x=[center_world[0, 0]],
            y=[center_world[1, 0]],
            z=[center_world[2, 0]],
            mode="markers",
            marker=dict(size=size, color=color),
            showlegend=False,
        )
    )
    
    # If an image is given, overlay it on the image plane
    if image is not None:
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be HxWx3 RGB array"

        # Normalize image values to [0, 1] if necessary
        if image.max() > 1:
            image = image / 255.0

        # Use Mesh3D with vertex colors from image corners
        # (Map image corners to triangle colors – simple version)
        img_h, img_w, _ = image.shape
        corner_colors = np.array([
            image[0, -1],     # top-right
            image[0, 0],      # top-left
            image[-1, 0],     # bottom-left
            image[-1, -1],    # bottom-right
        ])

        # Define two triangles from the four corners
        x, y, z = corners_world
        i = [0, 1, 2]
        j = [1, 2, 3]
        k = [2, 3, 0]

        traces.append(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                facecolor=[
                    f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'
                    for c in [corner_colors[0], corner_colors[1]]
                ],
                showscale=False,
                opacity=1.0
            )
        )

    return traces


def visualize_nemo(images, encoder_results, decoder_results=None, camera_poses=None, conf:float=0.1,
                   feature_decomposition: Optional[Literal["pca"]]=None):
    global figure
    figure = None
    # Images
    prediction_image = get_prediction_image_from_decoder_output(images, decoder_results)[0]
    # Point Cloud Images
    NeMO_colors=None
    if feature_decomposition == 'pca':
        # use PCA to visualize NeMO features on the NeMO points
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import minmax_scale
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(encoder_results["features_3d"][0].cpu())
        scaled_pca_feat_post = minmax_scale(features_pca)
        NeMO_colors = scaled_pca_feat_post
    nemo_point_cloud_image = get_point_cloud_image(
        encoder_results["surface_points"].cpu().numpy(),
        height=prediction_image.shape[0], 
        color=NeMO_colors,
        axis_size=0.5, 
        marker_size=3,
    )[0]
    colors = rearrange(images[0], "bt c h w -> bt h w c")[decoder_results["conf"][0] > conf][None, ...]
    
    if decoder_results:
        xyz_point_cloud_image = get_point_cloud_image(
            decoder_results["pts3d"][0][decoder_results["conf"][0] > conf][None, ...].cpu().numpy(),
            height=prediction_image.shape[0],
            color=colors.cpu().numpy(),
            axis_size=0.5,
            marker_size=3,
            camera_poses=camera_poses,
            downsample=10_000,
            frustum_scale=0.05
        )[0]
        combined_image = np.concatenate([nemo_point_cloud_image, prediction_image, xyz_point_cloud_image], axis=1)
    else:
        combined_image = np.concatenate([nemo_point_cloud_image, prediction_image], axis=1)
    return combined_image


def get_point_cloud_image(
    pointcloud,
    color=None,
    width=None,
    height=None,
    marker_size: int = 1,
    figure=None,
    camera_poses: torch.Tensor = None,
    axis_size: float = None,
    camera_up = None,
    camera_center = None,
    camera_eye = None,
    aspectmode="cube",
    downsample: Optional[int]=None,
    camera_images: Optional=None,
    frustum_scale:float=0.1,
    camera_colors: Union[str, Sequence] = 'black',
):
    if width is None and height is None:
        width = 800
        height = 600
    elif height is None:
        height = width * 3 / 4
    elif width is None:
        width = height * 4 / 3
    points = pointcloud.reshape((-1, 3))
    color = color.reshape((-1, 3)) if color is not None else np.clip((points[:, ...] + 1) / 2, 0, 1)
    if downsample:
        random_ids = torch.randint(0, points.shape[0], (downsample,))
        points = points[random_ids]
        color = color[random_ids]

    max_distance = np.max(np.linalg.norm(points, axis=1))

    # Set fixed axis ranges to prevent auto-scaling
    axis_size = max_distance if axis_size is None else axis_size
    axis_ranges = [
        [-axis_size, +axis_size],  # X-axis
        [-axis_size, +axis_size],  # Y-axis
        [-axis_size, +axis_size],  # Z-axis
    ]

    if figure is None:
        figure = go.Figure()
        myticks = np.arange(-axis_size, axis_size+0.1, (axis_size)/10) #[-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        myticks = []#[round(x, 6) for x in myticks.tolist()]
        figure.update_layout(
            scene=dict(
                xaxis=dict(visible=True, title="X", range=axis_ranges[0], tickvals=myticks, ticktext=myticks),
                yaxis=dict(visible=True, title="Y", range=axis_ranges[1], tickvals=myticks, ticktext=myticks),
                zaxis=dict(visible=True, title="Z", range=axis_ranges[2], tickvals=myticks, ticktext=myticks),
                aspectmode=aspectmode,
                dragmode='orbit',
            ),
            # margin=dict(l=0, r=0, b=0, t=0),
            scene_camera=dict(
                up=camera_up if camera_up else dict(x=0, y=0, z=1),
                center=camera_center if camera_center else dict(x=0, y=0, z=0),
                eye=camera_eye if camera_eye else dict(x=2.0, y=2.0, z=2.0),
                # projection=dict(type="orthographic"),
            ),
        )
    figure.data = []
    
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color,
                opacity=1.0,
            ),
            showlegend=False,
        )
    )
    
    if camera_poses is not None:
        camera_poses = camera_poses.cpu()
        for i, cp in enumerate(camera_poses):
            R = cp[:3, :3]
            t = cp[:3, 3]
            cam_img = camera_images[i] if camera_images else None
            cam_c = camera_colors[i] if isinstance(camera_colors, Sequence) else camera_colors
            frustum = get_opencv_camera_frustum(R, t, scale=frustum_scale, size=5, color=cam_c, image=cam_img)
            for line in frustum:
                figure.add_trace(line)
    img_bytes = figure.to_image(format="png", width=width, height=height)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = np.array(image)   # shape (H, W, 3)

    return image, figure

def save_point_cloud_html(
    pointcloud,
    color=None,
    width=None,
    height=None,
    marker_size: int = 1,
    camera_poses: Optional[torch.Tensor] = None,
    axis_size: Optional[float] = None,
    camera_up=None,
    camera_center=None,
    camera_eye=None,
    aspectmode="cube",
    downsample: Optional[int] = None,
    camera_images: Optional = None,
    frustum_scale: float = 0.1,
    camera_colors: Union[str, Sequence] = 'black',
    html_file: str = "pointcloud.html",
):
    if width is None and height is None:
        width = 800
        height = 600
    elif height is None:
        height = width * 3 / 4
    elif width is None:
        width = height * 4 / 3

    points = pointcloud.reshape((-1, 3))
    color = color.reshape((-1, 3)) if color is not None else np.clip((points[:, ...] + 1) / 2, 0, 1)

    if downsample:
        random_ids = torch.randint(0, points.shape[0], (downsample,))
        points = points[random_ids]
        color = color[random_ids]

    max_distance = np.max(np.linalg.norm(points, axis=1))
    axis_size = max_distance if axis_size is None else axis_size
    axis_ranges = [
        [-axis_size, +axis_size],
        [-axis_size, +axis_size],
        [-axis_size, +axis_size],
    ]

    figure = go.Figure()
    myticks = []  # optional: customize tick labels if needed
    figure.update_layout(
        scene=dict(
            xaxis=dict(visible=True, title="X", range=axis_ranges[0], tickvals=myticks, ticktext=myticks),
            yaxis=dict(visible=True, title="Y", range=axis_ranges[1], tickvals=myticks, ticktext=myticks),
            zaxis=dict(visible=True, title="Z", range=axis_ranges[2], tickvals=myticks, ticktext=myticks),
            aspectmode=aspectmode,
            dragmode='orbit',
        ),
        scene_camera=dict(
            up=camera_up if camera_up else dict(x=0, y=0, z=1),
            center=camera_center if camera_center else dict(x=0, y=0, z=0),
            eye=camera_eye if camera_eye else dict(x=2.0, y=2.0, z=2.0),
        ),
    )

    # Add point cloud
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color,
                opacity=1.0,
            ),
            showlegend=False,
        )
    )

    # Optional: camera frustums
    if camera_poses is not None:
        camera_poses = camera_poses.cpu()
        for i, cp in enumerate(camera_poses):
            R = cp[:3, :3]
            t = cp[:3, 3]
            cam_img = camera_images[i] if camera_images else None
            cam_c = camera_colors[i] if isinstance(camera_colors, Sequence) else camera_colors
            frustum = get_opencv_camera_frustum(R, t, scale=frustum_scale, size=5, color=cam_c, image=cam_img)
            for line in frustum:
                figure.add_trace(line)

    # Save to HTML
    figure.write_html(html_file, include_plotlyjs='cdn')


def get_prediction_image(xyz_images, est_xyz, est_xyz_conf):
    batch_size, t, c, h, w = xyz_images.shape
    aaa = (rearrange(xyz_images, "b t c h w -> (b t h) w c")).detach().cpu().numpy()
    # bbb = ((rearrange(gt_xyz, "b h w c -> (b h) w c") + 1) / 2).detach().cpu().numpy()
    ccc = ((rearrange(est_xyz, "b t h w c -> (b t h) w c") + 1) / 2).detach().cpu().numpy()

    ddd = np.stack(confidence_to_imgs(rearrange(est_xyz_conf, "b t h w -> (b t) h w").detach().cpu().numpy(), "inf"))
    ddd = rearrange(ddd, "b h w c -> (b h) w c")
    output_images = np.concatenate([aaa, ccc, ddd], axis=1) * 255
    output_images = output_images.astype(np.uint8)
    return output_images

def get_prediction_image_full(xyz_images,
                                est_xyz,  # (b t) h w c
                                est_xyz_conf,  # (b t) h w
                                est_depth_scaled,  # (b t) h w
                                est_depth_scaled_conf,  # (b t) h w
                                est_mask,  # (b t) h w
                                est_mask_full,  # (b t) h w
                                show_depth: bool=False,
        ):
        batch_size, t, c, h, w = xyz_images.shape
        out = []
        # GT
        out.append((rearrange(xyz_images, "b t c h w -> (b t h) w c")).detach().cpu().numpy())
        # Est
        out.append(((rearrange(est_xyz, "b h w c -> (b h) w c") + 1) / 2).detach().cpu().numpy())
        # ---
        img = np.stack(confidence_to_imgs(est_xyz_conf.detach().cpu().numpy(), "inf"))
        out.append(rearrange(img, "b h w c -> (b h) w c"))
        # ---
        if show_depth:
            img = np.stack(confidence_to_imgs(est_depth_scaled.detach().cpu().numpy(), 1, colormap="plasma"))
            out.append(rearrange(img, "b h w c -> (b h) w c"))
            img = np.stack(confidence_to_imgs(est_depth_scaled_conf.detach().cpu().numpy(), "inf"))
            out.append(rearrange(img, "b h w c -> (b h) w c"))
        # ---
        img = np.stack(confidence_to_imgs(est_mask.detach().cpu().numpy()>0.5, 1))
        out.append(rearrange(img, "b h w c -> (b h) w c"))
        # ---
        img = np.stack(confidence_to_imgs(est_mask_full.detach().cpu().numpy()>0.5, 1))
        out.append(rearrange(img, "b h w c -> (b h) w c"))

        out = np.concatenate(out, axis=1) * 255
        out = out.astype(np.uint8)
        output_images = np.split(out, batch_size)
        
        return output_images
    
def get_prediction_image_from_decoder_output(xyz_images,
                                            decoder_output,
                                            show_depth: bool = False):
        est_xyz = rearrange(decoder_output["pts3d"], "b t h w c -> (b t) h w c")
        est_xyz_conf = rearrange(decoder_output["conf"], "b t h w -> (b t) h w")
        est_depth_scaled = rearrange(decoder_output["depth_scaled"], "b t h w -> (b t) h w")
        est_depth_scaled_conf = rearrange(decoder_output["depth_scaled_conf"], "b t h w -> (b t) h w")
        est_mask = rearrange(decoder_output["mask"], "b t h w -> (b t) h w")
        est_mask_full = rearrange(decoder_output["mask_full"], "b t h w -> (b t) h w")
        
        output_images = get_prediction_image_full(xyz_images,
                                                est_xyz,
                                                est_xyz_conf,
                                                est_depth_scaled,
                                                est_depth_scaled_conf,
                                                est_mask,
                                                est_mask_full,
                                                show_depth)
        return output_images