"""
Run NeMO model on live camera frames and show predictions in real time.
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
    camera_id = args.camera_id

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

        # Convert to tensor
        img_tensor = image_to_tensor(frame_rgb, device=device, normalize=False)
        img_tensor = rearrange(img_tensor, 't c h w -> 1 t c h w')  # shape [1,1,3,H,W]

        # Decode
        with torch.no_grad():
            decoder_output = model.decode_images(img_tensor, nemo["features_3d_updated"])
        vis = get_prediction_image_from_decoder_output(img_tensor, decoder_output)[0]

        # Convert back to BGR for OpenCV display
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

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
    args = parser.parse_args()
    main(args)
