#!/usr/bin/env python3
import argparse
import os
import cv2
import torch
from functools import partial
from einops import rearrange
import subprocess
import numpy as np
import platform
import webbrowser
import re
from pathlib import Path

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
    (0.000, 0.447, 0.741),
    (0.850, 0.325, 0.098),
    (0.929, 0.694, 0.125),
    (0.494, 0.184, 0.556),
    (0.466, 0.674, 0.188),
    (0.301, 0.745, 0.933),
    (0.635, 0.078, 0.184),
    (0.500, 0.500, 0.500),
    (0.666, 0.333, 0.000),
    (0.333, 0.333, 0.000),
    (0.000, 0.500, 0.500),
    (0.600, 0.600, 0.000),
    (0.000, 0.000, 0.000),
    (1.000, 0.000, 1.000),
    (0.502, 0.000, 0.502),
    (0.000, 0.000, 1.000),
    (1.000, 0.647, 0.000),
    (0.824, 0.706, 0.549),
    (0.118, 0.565, 1.000),
    (0.255, 0.412, 0.882),
]


# ---------------------------------------------------------
# Cross-platform file opener
# ---------------------------------------------------------
def open_file_in_browser(path: Path):
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        webbrowser.open(f"file://{path.resolve()}")


# ---------------------------------------------------------
# Cross-platform camera opener
# ---------------------------------------------------------
def open_camera(index):
    if platform.system() == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def main():
    parser = argparse.ArgumentParser(
        description="Unified NeMO Capture, Reconstruct, and Live Inference"
    )
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth")
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--camera-id", default=None, type=int)
    parser.add_argument("--conf-threshold", default=1.2, type=float)
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"--- Phase 1: Initializing Model on {device} ---")
    model = Model.from_checkpoint(args.checkpoint, device=device)
    model.eval()

    object_name = args.object_name or input("Enter the object name: ").strip()

    # Windows-safe filename sanitization
    object_name = re.sub(r'[<>:"/\\|?*]', "_", object_name)

    out_dir = Path(args.output_dir) / object_name
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Capture Phase
    # ---------------------------------------------------------
    print("--- Phase 2: Capturing Images (ENTER=save, q=finish) ---")

    if args.camera_id is None:
        cap = None
        for i in range(10):
            test_cap = open_camera(i)
            if test_cap.isOpened():
                print(f"Camera found at index {i}")
                cap = test_cap
                camera_id = i
                break
            test_cap.release()
        if cap is None:
            print("No camera found.")
            return
    else:
        camera_id = args.camera_id
        cap = open_camera(camera_id)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    captured_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = center_crop(frame)
        display = frame.copy()
        cv2.putText(display, f"Captured: {len(captured_paths)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Capture Feed", display)

        key = cv2.waitKey(1)
        if key == 13:  # ENTER
            img_path = image_dir / f"{len(captured_paths):03d}.jpg"
            cv2.imwrite(str(img_path), frame)
            captured_paths.append(str(img_path))
            print(f"Saved: {img_path}")
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not captured_paths:
        print("No images captured.")
        return

    # ---------------------------------------------------------
    # Encode NeMO
    # ---------------------------------------------------------
    print("--- Phase 3: Generating Neural Memory Object (NeMO) ---")

    images_tensor = load_images(captured_paths, device=device, normalize=False)
    images_tensor = rearrange(images_tensor, 't c h w -> 1 t c h w')

    sample_points = torch.rand(1, 1500, 3, device=device) * 2 - 1

    with torch.no_grad():
        nemo = model.encode_images(images_tensor, sample_points)
        nemo["features_3d_updated"] = (
            nemo["features_3d"] + model.point_encoder(nemo["surface_points"])
        )
        nemo['scale_factor'] = None

    save_nemo(nemo, str(out_dir / "NeMO.pt"))
    print(f"NeMO saved to {out_dir}")

    # ---------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------
    print("--- Phase 4: Creating Interactive Reconstruction ---")

    with torch.no_grad():
        decoder_output = model.decode_images(
            images_tensor, nemo['features_3d_updated']
        )

    pts3d = decoder_output["pts3d"]
    conf = decoder_output["conf"]
    mask = conf[0] > args.conf_threshold

    pred_xyz = pts3d[0][mask][None, ...]
    bbox_3d = get_3d_bbox(pred_xyz[0])

    bbox_center = (bbox_3d[:3] + bbox_3d[3:]) / 2
    bbox_size = (bbox_3d[3:] - bbox_3d[:3]).max()

    pred_xyz[0] = (pred_xyz[0] - bbox_center) / bbox_size
    pred_colors = rearrange(
        images_tensor[0], "bt c h w -> bt h w c"
    )[mask][None, ...]

    # PnP
    print("Running PnP...")
    fake_intrinsic = torch.tensor(
        [[1000, 0, 112], [0, 1000, 112], [0, 0, 1]]
    ).unsqueeze(0).repeat(images_tensor.shape[1], 1, 1)[None].cpu().numpy()

    new_bbox = torch.tensor([[0, 0, 224, 224]]).repeat(
        images_tensor.shape[1], 1
    )[:, None, :]

    pnp_func = partial(
        model.forward,
        "pnp_pose_estimation",
        conf_threshold=args.conf_threshold,
        iterationsCount=1000,
        reprojectionError=6
    )

    pose_estimations = pnp_func(
        decoder_output, new_bbox, fake_intrinsic, nemo_scale_factor=1.0
    )

    # Filter valid poses
    pose_estimations = [p for p in pose_estimations if p is not None]

    # Ensure everything is a torch tensor on CPU
    pose_tensors = []
    for p in pose_estimations:
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p)
        pose_tensors.append(p.float())

    # Invert poses
    camera_poses = torch.stack(
        [torch.linalg.inv(p) for p in pose_tensors]
    )

    anchor_translation = camera_poses[0, :3, 3]
    camera_poses[:, :3, 3] -= bbox_center.cpu()
    camera_poses[:, :3, 3] /= (
        torch.linalg.norm(anchor_translation) + 1e-6
    )

    html_path = out_dir / "reconstruction.html"

    save_point_cloud_html(
        pred_xyz.cpu().numpy(),
        color=pred_colors.cpu().numpy(),
        height=1024,
        width=1024,
        axis_size=1.0,
        marker_size=1,
        downsample=5000,
        camera_poses=camera_poses,
        frustum_scale=0.1,
        camera_colors=camera_colors,
        html_file=str(html_path)
    )

    print(f"Visualization saved: {html_path}")
    open_file_in_browser(html_path)

    # ---------------------------------------------------------
    # Live Inference
    # ---------------------------------------------------------
    if args.skip_live:
        return

    print("--- Phase 5: Live Inference (q=quit) ---")

    cap = open_camera(camera_id)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            center_crop(frame_bgr), cv2.COLOR_BGR2RGB
        )
        img_t = image_to_tensor(
            frame_rgb, device=device, normalize=False
        ).unsqueeze(0)

        with torch.no_grad():
            dec_out = model.decode_images(
                img_t, nemo["features_3d_updated"]
            )
            poses = pnp_func(
                dec_out, new_bbox[:1], fake_intrinsic[:1],
                nemo_scale_factor=1.0
            )
            poses = [p for p in poses if p is not None]

        vis_rgb = get_prediction_image_from_decoder_output(
            img_t, dec_out
        )[0]
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        for pose in poses:
            vis_bgr = draw_pose_axes(
                vis_bgr, pose, fake_intrinsic[0, 0]
            )
            vis_bgr = draw_3d_box(
                vis_bgr, pose, fake_intrinsic[0, 0], scale=1.0
            )

        cv2.imshow("NeMO Live Inference", vis_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Pipeline Complete.")
    print(f"Output saved to {out_dir}")


if __name__ == "__main__":
    main()
