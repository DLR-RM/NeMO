"""Simple example showing the encoding of a NeMO."""
import argparse
from einops import rearrange
from nemolib.model import Model
import glob
from nemolib.visualization import get_point_cloud_image
from nemolib.utils import load_images, save_nemo
import cv2
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import os


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path: str = args.checkpoint
    image_paths = sorted(glob.glob(args.image_glob))
    print(image_paths)

    # --------------------------------------------------------------
    # Extract object name (parent folder of the images)
    # Example: assets/mug/*.jpg -> "mug"
    # --------------------------------------------------------------
    # Remove wildcard from glob, get parent folder name
    pattern_dir = os.path.dirname(args.image_glob)
    object_name = os.path.basename(pattern_dir)

    # Create output directory: out/<object_name>
    out_dir = os.path.join("out", object_name)
    os.makedirs(out_dir, exist_ok=True)
    # --------------------------------------------------------------

    images = load_images(image_paths, device=device, normalize=False)
    images = rearrange(images, 't c h w -> 1 t c h w')
    print(f"Found `{len(image_paths)}` images for object `{object_name}`.")
    model = Model.from_checkpoint(ckpt_path, device=device)
    model.eval()

    sample_points = torch.rand(1, 1500, 3, device=device) * 2 - 1
    with torch.no_grad():
        nemo = model.encode_images(images, sample_points)

    # PCA visualization
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(nemo["features_3d"][0].cpu())
    scaled_pca_feat_post = minmax_scale(features_pca)
    NeMO_colors = scaled_pca_feat_post

    nemo_point_cloud_image = get_point_cloud_image(
        nemo["surface_points"].cpu().numpy(),
        height=512,
        color=NeMO_colors,
        axis_size=0.7,
        marker_size=3,
        camera_up=dict(x=0, y=1, z=0),
        camera_eye=dict(x=-0.8, y=-0.8, z=-0.8),
    )[0]

    cv2.imwrite(os.path.join(out_dir, "NeMO_pca.png"), nemo_point_cloud_image)

    # Update features and remove scale
    nemo["features_3d_updated"] = nemo["features_3d"] + model.point_encoder(
        nemo["surface_points"]
    )
    nemo['scale_factor'] = None

    print(f"Saving Neural Memory Object to `{out_dir}`")
    save_nemo(nemo, os.path.join(out_dir, "NeMO.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint.pth",
        type=str,
        help="Checkpoint to test",
    )
    parser.add_argument("--image-glob", default='assets/mug/*.jpg', type=str)
    args = parser.parse_args()
    main(args)
