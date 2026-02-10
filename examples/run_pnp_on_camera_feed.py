"""
Run NeMO model on live camera frames and show predictions in real time.
"""
import argparse
from functools import partial
from einops import rearrange
from nemolib.model import Model
from nemolib.visualization import get_prediction_image_from_decoder_output, draw_3d_box, draw_pose_axes
from nemolib.utils import load_nemo, image_to_tensor, center_crop
import cv2
import numpy as np
import torch


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    nemo_path = args.nemo_path
    camera_id = args.camera_id
    conf_thr = args.conf_threshold
    # Load NeMO and model
    nemo = load_nemo(nemo_path, device=device)
    model = Model.from_checkpoint(ckpt_path, device=device)
    model.eval()

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera: {camera_id}")

    print("Press 'q' to quit.")
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = center_crop(frame_rgb)

        # Convert to tensor
        img_tensor = image_to_tensor(frame_rgb, device=device, normalize=False)
        img_tensor = rearrange(img_tensor, 't c h w -> 1 t c h w')  # shape [1,1,3,H,W]

        # Decode
        with torch.no_grad():
            decoder_output = model.decode_images(img_tensor, nemo["features_3d_updated"])
            
            
        # Define a partial function for PnP pose estimation.
        fake_intrinsic = torch.tensor([[1000, 0, 224 / 2], [0, 1000, 224 / 2], [0, 0, 1]])
        fake_intrinsic = fake_intrinsic.unsqueeze(0).repeat(img_tensor.shape[1], 1, 1)[None, ...].cpu().numpy()
        new_bbox = torch.tensor([[0,0,224,224]]).repeat(img_tensor.shape[1], 1)[:, None, :]
        pnp_func = partial(
            model.forward,
            "pnp_pose_estimation",
            conf_threshold=conf_thr,
            reprojectionError=2.0,
            confidence=0.99999,
            iterationsCount=500,
            min_inlier_ratio=0.0,
            verbose=False,
        )
        
        pose_estimations = pnp_func(decoder_output, new_bbox, fake_intrinsic, nemo_scale_factor=1.0)
        pose_estimations = [pose for pose in pose_estimations if pose is not None]
        object_poses = pose_estimations
        vis = get_prediction_image_from_decoder_output(img_tensor, decoder_output)[0]

        # Convert back to BGR for OpenCV display
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        print("Camera Poses", len(object_poses))
        if len(object_poses) > 0:
            print(object_poses)
            for pose_cam in object_poses:
                vis_bgr = draw_pose_axes(
                    vis_bgr,
                    pose_cam,
                    fake_intrinsic[0,0]
                )
                vis_bgr = draw_3d_box(
                    vis_bgr,
                    pose_cam,
                    fake_intrinsic[0,0],
                    scale=1.0
                )

        # Show input and prediction side by side
        combined = vis_bgr
        cv2.imshow("Camera | NeMO Prediction", combined)

        frame_idx += 1

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth", type=str)
    parser.add_argument("--nemo-path", default="out/mug/NeMO.pt", type=str)
    parser.add_argument("--camera-id", default=0, type=int, help="Camera device ID")
    parser.add_argument("--conf-threshold", default=1.2, type=float)
    args = parser.parse_args()
    main(args)
