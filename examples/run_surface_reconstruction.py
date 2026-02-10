"""Simple example showing the decoding of an image with a given NeMO."""
import argparse
from functools import partial
from einops import rearrange
from nemolib.model import Model
from nemolib.utils import load_images, load_nemo, get_3d_bbox, rodrigues_rotation, batch_transform_points
from nemolib.visualization import get_point_cloud_image
import glob
import cv2
import torch
import os
import numpy as np


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path: str = args.checkpoint
    image_paths = sorted(glob.glob(args.image_glob))
    nemo_path = args.nemo_path
    conf_thr = args.conf_threshold
    num_rotations = 36
    batch_size = 1
    theta = torch.zeros(batch_size)
    rotation_axis = torch.zeros(batch_size, 3)
    rotation_axis[..., 1] = 1  # y axis

    camera_colors = [
        (0.000, 0.447, 0.741),  # blue
        (0.850, 0.325, 0.098),  # orange
        (0.929, 0.694, 0.125),  # yellow
        (0.494, 0.184, 0.556),  # purple
        (0.466, 0.674, 0.188),  # green
        (0.301, 0.745, 0.933),  # cyan
        (0.635, 0.078, 0.184),  # red
        (0.500, 0.500, 0.500),  # gray
        (0.666, 0.333, 0.000),  # brown
        (0.333, 0.333, 0.000),  # olive
        (0.000, 0.500, 0.500),  # teal
        (0.600, 0.600, 0.000),  # mustard
        (0.000, 0.000, 0.000),  # black
        (1.000, 0.000, 1.000),  # magenta
        (0.502, 0.000, 0.502),  # dark magenta
        (0.000, 0.000, 1.000),  # deep blue
        (1.000, 0.647, 0.000),  # orange (light)
        (0.824, 0.706, 0.549),  # tan
        (0.118, 0.565, 1.000),  # dodger blue
        (0.255, 0.412, 0.882),  # royal blue
    ]

    # --------------------------------------------------------------
    # Extract object name from nemo_path
    # Example: out/mug/NeMO.pt -> "mug"
    # --------------------------------------------------------------
    nemo_dir = os.path.dirname(nemo_path)         # out/mug
    object_name = os.path.basename(nemo_dir)      # mug

    # Output directory
    out_dir = os.path.join("out", object_name)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "surface_reconstruction.mp4")
    # --------------------------------------------------------------
    fps = num_rotations / 3
    width = 512
    height = 512

    # Output video writer
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    images = load_images(image_paths, device=device, normalize=False)
    images = rearrange(images, 't c h w -> 1 t c h w')
    nemo = load_nemo(nemo_path, device=device)
    print(f"Found `{len(image_paths)}` images for object `{object_name}`.")
    
    model = Model.from_checkpoint(ckpt_path, device=device)
    model.eval()
    with torch.no_grad():
        decoder_output = model.decode_images(images, nemo['features_3d_updated'])
    pts3d = decoder_output["pts3d"]
    conf = decoder_output["conf"]

    predicted_xyz_points = pts3d[0].clone()[conf[0] > conf_thr][None, ...]  #
    bbox_3d =  get_3d_bbox(predicted_xyz_points[0])
    bbox_largest_size = (bbox_3d[3:] - bbox_3d[:3]).max()
    bbox_center = (bbox_3d[:3] + bbox_3d[3:]) / 2
    predicted_xyz_points[0] -= bbox_center
    predicted_xyz_points[0] /= bbox_largest_size
    predicted_xyz_colors = rearrange(images[0], "bt c h w -> bt h w c")[conf[0] > conf_thr][None, ...]
    
    # Define a partial function for PnP pose estimation.
    fake_intrinsic = torch.tensor([[1000, 0, 224 / 2], [0, 1000, 224 / 2], [0, 0, 1]])
    fake_intrinsic = fake_intrinsic.unsqueeze(0).repeat(images.shape[1], 1, 1)[None, ...].cpu().numpy()
    new_bbox = torch.tensor([[0,0,224,224]]).repeat(images.shape[1], 1)[:, None, :]
    pnp_func = partial(
        model.forward,
        "pnp_pose_estimation",
        conf_threshold=conf_thr,
        iterationsCount=1000,
        reprojectionError=6,
    )
    
    pose_estimations = pnp_func(decoder_output, new_bbox, fake_intrinsic, nemo_scale_factor=1.0)
    pose_estimations = torch.tensor(np.array([pose for pose in pose_estimations if pose is not None]))
    camera_poses = torch.tensor([torch.linalg.inv(T).detach().cpu().numpy() for T in pose_estimations])
    anchor_translation = camera_poses[0, :3, 3]
    camera_poses[:, :3, 3] -= bbox_center.cpu()
    camera_poses[:, :3, 3] = camera_poses[:, :3, 3] / (torch.linalg.norm(anchor_translation)+1e-6)
    global figure
    figure = None

    for x in range(num_rotations):
        print(f"({x + 1}/{num_rotations})")
        augmentation_trafo = torch.eye(4).repeat(batch_size, 1, 1)
        augmentation_trafo[..., :3, :3] = rodrigues_rotation(rotation_axis, theta)

        
        predicted_xyz_points_rotated = batch_transform_points(predicted_xyz_points.clone(), augmentation_trafo)
        R = augmentation_trafo[0, :3, :3]
        R_full = torch.eye(4, device=R.device, dtype=R.dtype).expand(camera_poses.shape[0], -1, -1).clone()
        R_full[:, :3, :3] = R  # Same R for all B
        camera_poses_rotated = camera_poses.clone().type(torch.float32)
        camera_poses_rotated = torch.bmm(R_full, camera_poses_rotated)
        
        # Output
        xyz_point_cloud_image = get_point_cloud_image(
            predicted_xyz_points_rotated.cpu().numpy(),
            color=predicted_xyz_colors.cpu().numpy(),
            height=height,
            width=width,
            axis_size=1.0,
            marker_size=1,
            camera_up=dict(x=0, y=-1, z=0),
            camera_eye=dict(x=0.8, y=-0.8, z=-0.8),
            aspectmode='cube',
            camera_poses=camera_poses_rotated,
            frustum_scale=0.1,
            camera_colors=camera_colors
        )[0]

        combined_image = np.concatenate([xyz_point_cloud_image], axis=1)
        vis_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        writer.write(vis_bgr)
        
        theta += 2 * torch.pi * (1 / num_rotations)
        
    writer.release()
    print(f"Done! Saved output to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint.pth",
        type=str,
        help="Checkpoint to test",
    )
    parser.add_argument(
        "--nemo-path",
        default="out/mug/NeMO.pt",
        type=str,
        help="Path to a NeMO",
    )
    parser.add_argument("--image-glob", default='assets/mug/*.jpg', type=str)
    parser.add_argument("--conf-threshold", default=1.2, type=float)
    args = parser.parse_args()
    main(args)
