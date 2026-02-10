#!/usr/bin/env python3
import argparse
import os
import cv2
import torch
from functools import partial
from einops import rearrange
import subprocess
import numpy as np

from nemolib.model import Model
from nemolib.visualization import (
    save_point_cloud_html,
    get_prediction_image_from_decoder_output,
    draw_3d_box,
    draw_pose_axes
)
from nemolib.utils import (
    save_nemo, load_images, center_crop,
    get_3d_bbox, image_to_tensor
)


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

def main():
    parser = argparse.ArgumentParser(description="Unified NeMO Capture, Reconstruct, and Live Inference")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth", type=str, help="Path to model checkpoint. If None, use default.")
    parser.add_argument("--object-name", default=None, type=str, help="Object name for output folder")
    parser.add_argument("--output-dir", default="out", type=str, help="Base output directory")
    parser.add_argument("--camera-id", default=None, type=int, help="Camera device ID")
    parser.add_argument("--conf-threshold", default=1.2, type=float, help="Confidence threshold")
    parser.add_argument("--skip-live", action="store_true", help="Skip the live inference phase")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. SETUP & MODEL LOADING (Only once)
    print(f"--- Phase 1: Initializing Model on {device} ---")
    model = Model.from_checkpoint(args.checkpoint, device=device)
    model.eval()

    object_name = args.object_name or input("Enter the object name: ").strip()
    out_dir = os.path.join(args.output_dir, object_name)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # 2. CAPTURE PHASE
    print(f"--- Phase 2: Capturing Images (Press ENTER to save, 'q' to finish) ---")
    # Open camera feed
    if not args.camera_id:
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera found at index {i}")
                camera_id = i
                break
            else:
                print(f"No camera at index {i}")
    else:
        camera_id = args.camera_id
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    captured_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = center_crop(frame)

        display = frame.copy()
        cv2.putText(display, f"Captured: {len(captured_paths)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Feed", display)

        key = cv2.waitKey(1)
        if key == 13: # ENTER
            img_path = os.path.join(image_dir, f"{len(captured_paths):03d}.jpg")
            cv2.imwrite(img_path, frame)
            captured_paths.append(img_path)
            print(f"Saved: {img_path}")
        elif key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    if not captured_paths:
        print("No images captured. Exiting.")
        return

    # 3. ENCODING PHASE (Generate NeMO once)
    print("--- Phase 3: Generating Neural Memory Object (NeMO) ---")
    images_tensor = load_images(captured_paths, device=device, normalize=False)
    images_tensor = rearrange(images_tensor, 't c h w -> 1 t c h w')

    sample_points = torch.rand(1, 1500, 3, device=device) * 2 - 1
    with torch.no_grad():
        nemo = model.encode_images(images_tensor, sample_points)
        # Update features immediately
        nemo["features_3d_updated"] = nemo["features_3d"] + model.point_encoder(nemo["surface_points"])
        nemo['scale_factor'] = None

    save_nemo(nemo, os.path.join(out_dir, "NeMO.pt"))
    print(f"NeMO saved to {out_dir}")

    # 4. STATIC RECONSTRUCTION PHASE (With Camera Visualization)
    print("--- Phase 4: Creating Interactive Reconstruction with Cameras ---")
    with torch.no_grad():
        decoder_output = model.decode_images(images_tensor, nemo['features_3d_updated'])

    # 4a. Point Cloud Processing
    pts3d = decoder_output["pts3d"]
    conf = decoder_output["conf"]
    mask = conf[0] > args.conf_threshold

    pred_xyz = pts3d[0].clone()[mask][None, ...]
    bbox_3d = get_3d_bbox(pred_xyz[0])
    bbox_center = (bbox_3d[:3] + bbox_3d[3:]) / 2
    bbox_size = (bbox_3d[3:] - bbox_3d[:3]).max()

    pred_xyz[0] = (pred_xyz[0] - bbox_center) / bbox_size
    pred_colors = rearrange(images_tensor[0], "bt c h w -> bt h w c")[mask][None, ...]

    # 4b. Camera Pose Estimation (PnP)
    fake_intrinsic = torch.tensor([[1000, 0, 224 / 2], [0, 1000, 224 / 2], [0, 0, 1]])
    fake_intrinsic = fake_intrinsic.unsqueeze(0).repeat(images_tensor.shape[1], 1, 1)[None, ...].cpu().numpy()
    new_bbox = torch.tensor([[0,0,224,224]]).repeat(images_tensor.shape[1], 1)[:, None, :]

    pnp_func_static = partial(
        model.forward, "pnp_pose_estimation",
        conf_threshold=args.conf_threshold, iterationsCount=1000, reprojectionError=6
    )


    # Filter valid poses and transform to Camera-to-World
    print("Running pnp...")
    pose_estimations = pnp_func_static(decoder_output, new_bbox, fake_intrinsic, nemo_scale_factor=1.0)
    pose_estimations = torch.tensor(np.array([pose for pose in pose_estimations if pose is not None]))
    # Convert all poses to numpy arrays first
    poses_np = np.array([torch.linalg.inv(T).detach().cpu().numpy() for T in pose_estimations])
    # Then convert the whole array to a single tensor
    camera_poses = torch.from_numpy(poses_np)
    camera_poses = torch.tensor([torch.linalg.inv(T).detach().cpu().numpy() for T in pose_estimations])
    anchor_translation = camera_poses[0, :3, 3]
    camera_poses[:, :3, 3] -= bbox_center.cpu()
    camera_poses[:, :3, 3] = camera_poses[:, :3, 3] / (torch.linalg.norm(anchor_translation)+1e-6)
    print("PnP Done.")

    html_path = os.path.join(out_dir, "reconstruction.html")

    print("Generating Visualization")
    save_point_cloud_html(
        pred_xyz.cpu().numpy(),
        color=pred_colors.cpu().numpy(),
        height=1024,
        width=1024,
        axis_size=1.0,
        marker_size=1,
        downsample=5_000,
        camera_up=dict(x=0, y=-1, z=0),
        camera_eye=dict(x=0.8, y=-0.8, z=-0.8),
        aspectmode='cube',
        camera_poses=camera_poses,
        frustum_scale=0.1,
        camera_colors=camera_colors,
        html_file=html_path
    )
    print("Visualisation done.")

    print(f"HTML Visualization saved: {html_path}")
    subprocess.Popen(["xdg-open", html_path])

    # 5. LIVE INFERENCE PHASE
    if args.skip_live:
        return

    print("--- Phase 5: Starting Live Real-Time Inference (Press 'q' to quit) ---")
    cap = cv2.VideoCapture(camera_id)

    fake_intrinsic = torch.tensor([[1000, 0, 224 / 2], [0, 1000, 224 / 2], [0, 0, 1]])
    fake_intrinsic = fake_intrinsic.unsqueeze(0).repeat(1, 1, 1)[None, ...].cpu().numpy()
    new_bbox = torch.tensor([[0,0,224,224]]).repeat(1, 1)[:, None, :]


    pnp_func = partial(
        model.forward,
        "pnp_pose_estimation",
        conf_threshold=args.conf_threshold,
        reprojectionError=2.0,
        confidence=0.99999,
        min_inlier_ratio=0.0,
        iterationsCount=500,
        verbose=False
    )

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(center_crop(frame_bgr), cv2.COLOR_BGR2RGB)
        img_t = image_to_tensor(frame_rgb, device=device, normalize=False).unsqueeze(0)

        with torch.no_grad():
            dec_out = model.decode_images(img_t, nemo["features_3d_updated"])
            poses = pnp_func(dec_out, new_bbox, fake_intrinsic, nemo_scale_factor=1.0)
            poses = [p for p in poses if p is not None]

        vis_rgb = get_prediction_image_from_decoder_output(img_t, dec_out)[0]
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        for pose in poses:
            vis_bgr = draw_pose_axes(vis_bgr, pose, fake_intrinsic[0, 0])
            vis_bgr = draw_3d_box(vis_bgr, pose, fake_intrinsic[0, 0], scale=1.0)

        cv2.imshow("NeMO Live Inference, press `q` to exit.", vis_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline Complete.")
    print(f"Output saved to {out_dir}")
    print(f"To see the surface reconstruction in the browser, open `{html_path}`")

if __name__ == "__main__":
    main()
