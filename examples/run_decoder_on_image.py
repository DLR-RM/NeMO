"""Simple example showing the decoding of an image with a given NeMO."""
import argparse
from einops import rearrange
from nemolib.model import Model
from nemolib.visualization import get_prediction_image_from_decoder_output
from nemolib.utils import load_image, load_nemo
import cv2
import torch
import os


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path: str = args.checkpoint
    image_path = args.image_path
    nemo_path = args.nemo_path

    # --------------------------------------------------------------
    # Extract object name from nemo_path
    # Example: out/mug/NeMO.pt -> "mug"
    # --------------------------------------------------------------
    nemo_dir = os.path.dirname(nemo_path)         # out/mug
    object_name = os.path.basename(nemo_dir)      # mug

    # Output directory
    out_dir = os.path.join("out", object_name)
    os.makedirs(out_dir, exist_ok=True)
    # --------------------------------------------------------------

    images = load_image(image_path, device=device, normalize=False)
    images = rearrange(images, 't c h w -> 1 t c h w')
    nemo = load_nemo(nemo_path, device=device)

    model = Model.from_checkpoint(ckpt_path, device=device)
    model.eval

    with torch.no_grad():
        decoder_output = model.decode_images(images, nemo['features_3d_updated'])
    print('##### Decoder Output Keys:')
    for key in decoder_output:
        print(f"\t- {key}")

    dec_out_vis = get_prediction_image_from_decoder_output(images, decoder_output)[0]

    out_path = os.path.join(out_dir, "decoder_output.png")
    cv2.imwrite(out_path, cv2.cvtColor(dec_out_vis, cv2.COLOR_RGB2BGR))

    print(f"Written decoder output to `{out_path}`")


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
    parser.add_argument("--image-path", default='assets/query_image.jpg', type=str)
    args = parser.parse_args()
    main(args)
