"""
Run multiple NeMO models on live camera frames and show predictions in real time.
Each NeMO prediction is shown in its own row.
"""
import argparse
from einops import rearrange
from nemolib.model import Model
from nemolib.visualization import get_prediction_image_from_decoder_output
from nemolib.utils import load_nemo, image_to_tensor, center_crop
import cv2
import torch
import os

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
    """
    Load multiple NeMO features and stack into shape [N, 1500, 768].
    Each NeMO.pt is assumed to contain features_3d_updated of shape [1, 1500, 768].
    """
    features_list = []
    for nemo_name in nemo_names:
        nemo_path = os.path.join(base_path, nemo_name, "NeMO.pt")
        nemo = load_nemo(nemo_path, device=device)
        # Remove batch dim: [1,1500,768] -> [1500,768]
        features = nemo["features_3d_updated"].squeeze(0)
        features_list.append(features)
    
    # Stack along new batch dimension: [N, 1500, 768]
    features_batch = torch.stack(features_list, dim=0)
    return features_batch

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ask which NeMOs to load
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
            # Repeat image tensor along batch dimension to match number of NeMOs
            batch_img_tensor = img_tensor.repeat(features_batch.shape[0], 1, 1, 1, 1)  # [B,1,3,H,W]
            decoder_output = model.decode_images(batch_img_tensor, features_batch)

        # Visualize each NeMO prediction
        vis_list = get_prediction_image_from_decoder_output(batch_img_tensor, decoder_output)

        # Stack vertically
        combined_vis = cv2.vconcat([cv2.cvtColor(v, cv2.COLOR_RGB2BGR) for v in vis_list])
        cv2.imshow("Camera | NeMO Predictions", combined_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth", type=str)
    parser.add_argument("--camera-id", default=0, type=int, help="Camera device ID")
    args = parser.parse_args()
    main(args)
