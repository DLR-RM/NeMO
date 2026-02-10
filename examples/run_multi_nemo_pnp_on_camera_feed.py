"""
Run multiple NeMO models on live camera frames with PnP pose estimation.
Each NeMO object is visualized in a single video feed with its own color and label.
"""
import argparse
from functools import partial
from einops import rearrange
from nemolib.model import Model
from nemolib.visualization import draw_3d_box, draw_pose_axes
from nemolib.utils import load_nemo, image_to_tensor, center_crop
import cv2
import torch
import numpy as np
import os

# Predefined colors for objects
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]

def select_nemos(base_path="out"):
    """Ask user which NeMOs to load."""
    nemo_options = [d for d in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, d, "NeMO.pt")) or
                    os.path.isdir(os.path.join(base_path, d))]
    if not nemo_options:
        raise ValueError(f"No NeMO models found in {base_path}")

    print("Available NeMOs:")
    for idx, nemo_name in enumerate(nemo_options):
        print(f"{idx}: {nemo_name}")
    
    selection = input("Enter comma-separated indices of NeMOs to load (e.g., 0,2): ")
    indices = [int(s.strip()) for s in selection.split(",")]
    selected_nemos = [nemo_options[i] for i in indices]
    return selected_nemos

def load_nemo_features(nemo_names, device, base_path="out"):
    """Load multiple NeMO features and stack into shape [N, 1500, 768]."""
    features_list = []
    for nemo_name in nemo_names:
        nemo_path = os.path.join(base_path, nemo_name, "NeMO.pt")
        nemo = load_nemo(nemo_path, device=device)
        features = nemo["features_3d_updated"].squeeze(0)
        features_list.append(features)
    features_batch = torch.stack(features_list, dim=0)
    return features_batch

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select NeMOs
    selected_nemos = select_nemos()
    features_batch = load_nemo_features(selected_nemos, device=device)

    # Load single model
    model = Model.from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera: {args.camera_id}")

    print("Press 'q' to quit.")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to RGB and tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = center_crop(frame_rgb)
        img_tensor = image_to_tensor(frame_rgb, device=device, normalize=False)
        img_tensor = rearrange(img_tensor, 't c h w -> 1 t c h w')  # [1,1,3,H,W]

        with torch.no_grad():
            # Repeat image for batch of NeMOs
            batch_img_tensor = img_tensor.repeat(features_batch.shape[0], 1, 1, 1, 1)  # [N,1,3,H,W]
            decoder_output = model.decode_images(batch_img_tensor, features_batch)

        # Define fake intrinsic and bounding boxes for PnP
        fake_intrinsic = torch.tensor([[1000, 0, 224/2], [0, 1000, 224/2], [0, 0, 1]])
        fake_intrinsic = fake_intrinsic.unsqueeze(0).repeat(features_batch.shape[0], 1, 1)[None, ...].cpu().numpy()

        new_bbox = torch.tensor([[0,0,224,224]]).repeat(features_batch.shape[0], 1)[:, None, :]

        # Partial function for PnP
        pnp_func = partial(
            model.forward,
            "pnp_pose_estimation",
            conf_threshold=args.conf_threshold,
            reprojectionError=2.0,
            confidence=0.99999,
            iterationsCount=500,
            min_inlier_ratio=0.0,
            verbose=False,
        )

        pose_estimations = pnp_func(decoder_output, new_bbox, fake_intrinsic, nemo_scale_factor=1.0)
        #pose_estimations = [pose for pose in pose_estimations if pose is not None]

        # Visualization
        vis_bgr = cv2.cvtColor(rearrange(img_tensor[0,0], 'c h w -> h w c').cpu().numpy(), cv2.COLOR_BGR2RGB)
        for i, pose_cam in enumerate(pose_estimations):
            if pose_cam is None: continue
            color = COLORS[i % len(COLORS)]
            vis_bgr = draw_pose_axes(vis_bgr, pose_cam, fake_intrinsic[0,0])
            vis_bgr = draw_3d_box(vis_bgr, pose_cam, fake_intrinsic[0,0], scale=1.0, color=color)
            # Add label
            cv2.putText(
                vis_bgr,
                selected_nemos[i],
                (10, 30 + i*25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Camera | Multi-NeMO PnP", vis_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth", type=str)
    parser.add_argument("--camera-id", default=0, type=int, help="Camera device ID")
    parser.add_argument("--conf-threshold", default=1.2, type=float)
    args = parser.parse_args()
    main(args)
