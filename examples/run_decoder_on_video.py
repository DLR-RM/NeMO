"""
Run NeMO model on every frame of a video and save the predictions as a new MP4.
"""
import argparse
from einops import rearrange
from nemolib.model import Model
from nemolib.visualization import get_prediction_image_from_decoder_output
from nemolib.utils import load_nemo, image_to_tensor
import cv2
import torch
import os


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    nemo_path = args.nemo_path
    video_path = args.video_path

    # --------------------------------------------------------------
    # Extract object name from nemo_path and create output folder
    # Example: out/mug/NeMO.pt -> "mug" -> out/mug/video_predictions.mp4
    # --------------------------------------------------------------
    nemo_dir = os.path.dirname(nemo_path)        # out/mug
    object_name = os.path.basename(nemo_dir)     # mug
    out_dir = os.path.join("out", object_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "video_predictions.mp4")
    # --------------------------------------------------------------

    # Load NeMO and model
    nemo = load_nemo(nemo_path, device=device)
    model = Model.from_checkpoint(ckpt_path, device=device)
    model.eval()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 1120 
    height = 224

    # Output video writer
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    print("Processing video...")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert to RGB for model
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        img_tensor = image_to_tensor(frame_rgb, device=device, normalize=False)
        img_tensor = rearrange(img_tensor, 't c h w -> 1 t c h w')  # shape [1,1,3,H,W]
        
        # Decode
        with torch.no_grad():
            decoder_output = model.decode_images(img_tensor, nemo["features_3d_updated"])
        vis = get_prediction_image_from_decoder_output(img_tensor, decoder_output)[0]

        # Back to BGR to write to output
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        writer.write(vis_bgr)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Done! Saved output to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth", type=str)
    parser.add_argument("--nemo-path", default="out/mug/NeMO.pt", type=str)
    parser.add_argument("--video-path", default="assets/mug/video.mp4", type=str)
    args = parser.parse_args()
    main(args)
