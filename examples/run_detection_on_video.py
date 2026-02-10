"""
Run NeMO model on every frame of a video and save the predictions as a new MP4.
"""
import argparse
import copy
from typing import Optional
from einops import rearrange
from nemolib.model import Model
from nemolib.utils import load_nemo, image_to_tensor, scale_images, masks_to_bounding_boxes
import cv2
import torch
import os
import math
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes


def segment_objects(model, features, image, mask_thr: float = 0.5, patch_size=128, stride=64, device: Optional[torch.device] = None):
    """
    Segment objects by sliding a patch-based predictor over the image.

    Args:
        model:
        features (torch.Tensor): Feature vectors of shape (B, N, C_feat).
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        mask_thr (float): Threshold for the confidence map.
        patch_size (int): Height and width of each patch (default: 128).
        stride (int): Stride for the sliding window (default: 64).

    Returns:
        mask (torch.Tensor): Binary mask of shape (B, 1, H, W) (or (B, H, W)) where each pixel is 1
                             if the averaged confidence > T.
        confidence_image (torch.Tensor): The averaged confidence map of shape (B, 1, H, W).
    """
    image = image.to(device)
    features = features.to(device)
    C, H, W = image.shape
    B, N, C_feat = features.shape

    # Compute number of patches needed along H and W
    nH = math.ceil((H - patch_size) / stride) + 1
    nW = math.ceil((W - patch_size) / stride) + 1
    total_h = (nH - 1) * stride + patch_size
    total_w = (nW - 1) * stride + patch_size
    pad_h = total_h - H  # extra rows needed at bottom
    pad_w = total_w - W  # extra cols needed at right

    # Pad image on right and bottom so that the sliding windows cover the entire image
    image_padded = F.pad(image, (0, pad_w, 0, pad_h))  # pad: (left, right, top, bottom)
    # Add batch dimension
    image_padded = image_padded[None, ...]
    # Use unfold to extract patches.
    # Resulting shape is (B, C * patch_size * patch_size, n_patches)
    patches = F.unfold(image_padded, kernel_size=(patch_size, patch_size), stride=(stride, stride))
    nP = patches.shape[-1]

    # Reshape to (1, nP, C, patch_size, patch_size)
    patches = patches.transpose(1, 2).reshape(1, nP, C, patch_size, patch_size)
    patches = patches.expand(B, -1, -1, -1, -1)  # B nP C patch_size patch_size

    # Run xyz prediction
    with torch.no_grad():
        xyz_results = model("xyz", images=patches, features_3d=features)
    # TODO: On 02.06.25 We changed the network to output sigmoid scaled mask directly! We do not need it here anymore
    amodal_mask = xyz_results["mask_full"]  # B nP 224 224, use sigmoid to map mask from 0 to 1
    modal_mask = xyz_results["mask"]
    # Rescale back to original patch size
    amodal_mask = scale_images(amodal_mask, patch_size, patch_size, mode="nearest", align_corners=None)
    modal_mask = scale_images(modal_mask, patch_size, patch_size, mode="nearest", align_corners=None)

    # Reshape back to (B, nP, 1, patch_size, patch_size)
    amodal_mask = amodal_mask.reshape(B, nP, 1, patch_size, patch_size)
    modal_mask = modal_mask.reshape(B, nP, 1, patch_size, patch_size)

    # Now, we need to reassemble the patches back into an image.
    # First, flatten the confidence maps to match fold's input requirements.
    # We want a tensor of shape (B, 1 * patch_size * patch_size, nP)
    amodal_mask_flat = amodal_mask.reshape(B, nP, patch_size * patch_size).transpose(1, 2)
    modal_mask_flat = modal_mask.reshape(B, nP, patch_size * patch_size).transpose(1, 2)

    # Use fold to reassemble the patches into an image of size (total_h, total_w)
    amodal_confidence = F.fold(
        amodal_mask_flat, output_size=(total_h, total_w), kernel_size=(patch_size, patch_size), stride=(stride, stride)
    )
    modal_confidence = F.fold(
        modal_mask_flat, output_size=(total_h, total_w), kernel_size=(patch_size, patch_size), stride=(stride, stride)
    )

    # Because patches overlap, create an overlap count map.
    ones = torch.ones_like(amodal_mask)
    ones_flat = ones.reshape(B, nP, patch_size * patch_size).transpose(1, 2)
    overlap_count = F.fold(
        ones_flat, output_size=(total_h, total_w), kernel_size=(patch_size, patch_size), stride=(stride, stride)
    )

    # Average the overlapping predictions.
    amodal_confidence = amodal_confidence / overlap_count
    modal_confidence = modal_confidence / overlap_count

    # Remove the padding to return to the original image size.
    amodal_confidence = amodal_confidence[:, :, :H, :W]
    modal_confidence = modal_confidence[:, :, :H, :W]

    # Threshold the confidence map to produce a binary mask.
    amodal_mask = (amodal_confidence > mask_thr).float()
    modal_mask = (modal_confidence > mask_thr).float()

    return amodal_mask, amodal_confidence, modal_mask, modal_confidence

colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (192, 0, 0),  # Crimson
        (0, 192, 0),  # Emerald
        (0, 0, 192),  # Indigo
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Rose
        (0, 255, 128),  # Mint
        (128, 255, 0),  # Lime
        (0, 128, 255),  # Azure
        (128, 0, 255),  # Violet
        (255, 0, 192),  # Fuchsia
        (192, 255, 0),  # Chartreuse
        (0, 255, 192),  # Aquamarine
        (255, 192, 0),  # Amber
        (192, 0, 255),  # Orchid
        (0, 192, 255),  # Sky Blue
        (255, 64, 64),  # Salmon
        (64, 255, 64),  # Light Green
        (64, 64, 255),  # Periwinkle
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (192, 0, 0),  # Crimson
        (0, 192, 0),  # Emerald
        (0, 0, 192),  # Indigo
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Rose
        (0, 255, 128),  # Mint
        (128, 255, 0),  # Lime
        (0, 128, 255),  # Azure
        (128, 0, 255),  # Violet
        (255, 0, 192),  # Fuchsia
        (192, 255, 0),  # Chartreuse
        (0, 255, 192),  # Aquamarine
        (255, 192, 0),  # Amber
        (192, 0, 255),  # Orchid
        (0, 192, 255),  # Sky Blue
        (255, 64, 64),  # Salmon
        (64, 255, 64),  # Light Green
        (64, 64, 255),  # Periwinkle
    ]


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
    out_path = os.path.join(out_dir, "video_detection_predictions.mp4")
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
    width = 1440
    height = 1440

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
        rgb_canvas = copy.copy(frame_rgb)
        # Convert to tensor
        img_tensor = torch.from_numpy(rearrange(frame_rgb, 'h w c -> c h w')).float().to(device) / 255.0
        
        patch_sizes = [
                min(img_tensor.shape[-2:]) // 1,
                min(img_tensor.shape[-2:]) // 2,
                min(img_tensor.shape[-2:]) // 3,
            ]
        batched_amodal_mask, batched_amodal_conf = [], []
        batched_modal_mask, batched_modal_conf = [], []
        overlayed_amodal_mask = torch.zeros((1, 1, img_tensor.shape[1], img_tensor.shape[2]), device=device)
        overlayed_amodal_conf = torch.zeros((1, 1, img_tensor.shape[1], img_tensor.shape[2]), device=device)
        overlayed_modal_mask = torch.zeros((1, 1, img_tensor.shape[1], img_tensor.shape[2]), device=device)
        overlayed_modal_conf = torch.zeros((1, 1, img_tensor.shape[1], img_tensor.shape[2]), device=device)
        for ps in patch_sizes:
                # patch_size = min(rgb.shape[-2:]) // 2  # max(rgb.shape[-2:])
                stride = ps // 3  # max(rgb.shape[-2:]) // 4  # 1
                amodal_mask, amodal_confidence, modal_mask, modal_confidence = segment_objects(
                    model,
                    nemo['features_3d_updated'],
                    img_tensor,
                    patch_size=ps,
                    stride=stride,
                    device=device,
                )
                overlayed_amodal_mask += amodal_mask
                overlayed_amodal_conf += amodal_confidence
                overlayed_modal_mask += modal_mask
                overlayed_modal_conf += modal_confidence
        # Scale by number of different scales applied
        overlayed_amodal_mask, overlayed_amodal_conf = (
            overlayed_amodal_mask / len(patch_sizes),
            overlayed_amodal_conf / len(patch_sizes),
        )
        overlayed_modal_mask, overlayed_modal_conf = (
            overlayed_modal_mask / len(patch_sizes),
            overlayed_modal_conf / len(patch_sizes),
        )

        overlayed_amodal_mask = overlayed_amodal_mask > 0.0
        overlayed_modal_mask = overlayed_modal_mask > 0.0

        batched_amodal_mask.append(overlayed_amodal_mask)
        batched_amodal_conf.append(overlayed_amodal_conf)
        batched_modal_mask.append(overlayed_modal_mask)
        batched_modal_conf.append(overlayed_modal_conf)
        
        ##############
        # Transform list into tensor
        batched_amodal_mask, batched_amodal_conf = torch.cat(batched_amodal_mask), torch.cat(batched_amodal_conf)
        batched_modal_mask, batched_modal_conf = torch.cat(batched_modal_mask), torch.cat(batched_modal_conf)
        
        # Use amodal mask to get bounding box
        bounding_boxes, bb_confidences, _ = masks_to_bounding_boxes(
            batched_amodal_mask, batched_amodal_conf, minimum_area=32*32
        )
        batched_flat_bb, batched_flat_conf, batched_flat_ids, batched_flat_segm = [], [], [], []
        for n in range(1):
            batched_flat_bb.extend(bounding_boxes[n])
            batched_flat_conf.extend(bb_confidences[n])
            batched_flat_ids.extend([0] * len(bounding_boxes[n]))
            segm_in_bb = []
            for bb in bounding_boxes[n]:
                x = torch.zeros(batched_modal_mask.shape[-2:])
                x[bb[1] : bb[3], bb[0] : bb[2]] = batched_modal_mask[n, 0][bb[1] : bb[3], bb[0] : bb[2]]
                segm_in_bb.append(x)

            batched_flat_segm.extend(segm_in_bb)
        for oid, bb, segm, c in zip(batched_flat_ids, batched_flat_bb, batched_flat_segm, batched_flat_conf):
            rgb_canvas = draw_bounding_boxes(
                        rearrange(torch.tensor(rgb_canvas), "h w c -> c h w"),
                        torch.tensor([[int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]]),
                        [f"{oid}: {c:.2f}"],
                        colors=colors[oid],
                        fill=False,
                        width=1,
                        font_size=20,
                    )
            rgb_canvas = rearrange(rgb_canvas, "c h w -> h w c").numpy()

        # Back to BGR to write to output
        vis_bgr = cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR)
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
