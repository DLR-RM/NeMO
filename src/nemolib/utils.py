"""Utility helpers for nemolib.

Includes checkpoint loading, image preprocessing and tensor helpers.
"""
from typing import Any, Optional, Sequence, Union
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

import cv2



def confidence_to_imgs(confidence: np.ndarray, expected_max: float, colormap: str = "jet"):
    """
    Creates a list of images with jet color map based on a np array of shape (b, h, w)

    Returns a list of length b with numpy.ndarrays of shape (h, w, 3) in RGB

    Image values will be between 0 and 1
    """
    assert len(confidence.shape) == 3, "confidence has to be of shape (b, h, w)"
    b, h, w = confidence.shape

    cmap = plt.get_cmap(colormap)
    confs = [c[0] for c in np.split(confidence, b)]  # List of len B with tensors of shape (H, W)
    # Map confidence based on mapping
    if expected_max == float("inf") or expected_max == "inf":
        confs_max = max([d.max() for d in confs]) + 1e-5  # TODO: Maybe unique max per confidence image?
    else:
        confs_max = expected_max
    confs = [cmap(d / confs_max)[..., :3] for d in confs]  # Map to RGB instead of RGBA
    return confs

def get_3d_bbox(point_cloud: torch.Tensor) -> torch.Tensor:
    """
    Computes the axis-aligned 3D bounding box of a point cloud.

    Args:
        point_cloud (torch.Tensor): Tensor of shape (N, 3), where N is the number of points.

    Returns:
        torch.Tensor: Tensor of shape (6,) containing [xmin, ymin, zmin, xmax, ymax, zmax].
    """
    # Ensure input is a 2D tensor with shape (N, 3)
    assert point_cloud.dim() == 2 and point_cloud.shape[1] == 3, "Input must have shape (N, 3)"

    # Compute min and max along each axis
    min_vals = point_cloud.min(dim=0).values  # (3,)
    max_vals = point_cloud.max(dim=0).values  # (3,)

    # Concatenate into a single tensor
    bbox = torch.cat([min_vals, max_vals])  # (6,)

    return bbox


def rodrigues_rotation(axis, theta):
    """
    Perform rotation using Rodrigues' rotation formula.

    Args:
    - axis (torch.Tensor): Axis of rotation (3D vector). Shape: [N, 3].
    - theta (torch.Tensor): Angle of rotation (in radians). Shape: [N].

    Returns:
    - rotation_matrix (torch.Tensor): 3x3 rotation matrix. Shape: [N, 3, 3].
    """
    # Normalize axis
    k = axis / torch.norm(axis, dim=-1, keepdim=True)

    # Compute skew-symmetric matrix
    K = torch.zeros(axis.shape[0], 3, 3, device=axis.device, dtype=axis.dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    # Compute rotation matrix using Rodrigues' formula
    rotation_matrix = (
        torch.eye(3, device=axis.device, dtype=axis.dtype).repeat(axis.shape[0], 1, 1)
        + torch.sin(theta)[..., None, None] * K
        + (1 - torch.cos(theta)[..., None, None]) * (K @ K)
    )

    return rotation_matrix



def batch_transform_points(
    points: torch.Tensor, transformation: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Applies transfomation (Rotation + Translation) and scale to a batch of points.

    Points is of shape B, N, 3 where B is the batchsize, N is the number of points per batch and 3 is the point dimension.
    Transformation is of shape B, 4, 4. It's a transformation matrix
    Scale is of shape B. It's a scale factor applied to the points per batch.

    Returns the transformed points of shape [B, N, 3]
    """
    # points shape = B, N, 3
    # transformation shape = B, 4, 4
    # scale shape = B
    assert points.dim() == 3 and points.shape[2] == 3, "`points` has to be of shape [B, N, 3]"
    b, n, _ = points.shape
    device = points.device

    points_transformed = points.clone()
    if transformation is not None:
        assert transformation.dim() == 3 and transformation.shape[-2:] == (
            4,
            4,
        ), "`transformation` has to be of shape [B, 4, 4]"
        assert transformation.shape[0] == b, "`transformation` batch size is not the same as `points` batch_size"
        # Add dummy dimension
        points_transformed = torch.cat([points_transformed, torch.ones((*points_transformed.shape[:-1], 1)).to(device)], dim=2)
        transformation = transformation.to(device)
        # Transpose points to shape [2, 4, 1500] to align for batch matrix multiplication
        points_transformed = points_transformed.transpose(1, 2)  # Shape [b, 4, n]
        points_transformed = torch.bmm(transformation, points_transformed)
        points_transformed = points_transformed.transpose(1, 2)  # Shape [b, n, 4]
        # Remove dummy dimension
        points_transformed = points_transformed[..., :3]
    if scale is not None:
        assert scale.dim() == 1, "`scale` has to be of shape [B]"
        assert scale.shape[0] == b, "`scale` batch size is not the same as `points` batch_size"

        scale = scale.to(device)
        points_transformed = scale[..., None, None] * points_transformed
    return points_transformed

def masks_to_bounding_boxes(
    masks: torch.Tensor,
    confidences: torch.Tensor,
    minimum_area: int = 0,
):
    """
    Convert a batch of binary masks to bounding boxes using OpenCV,
    and compute the mean confidence of each box.

    Args:
        masks (torch.Tensor): (B, 1, H, W) binary masks.
        confidences (torch.Tensor): (B, 1, H, W) confidence values in [0, 1].
        minimum_area (int): Minimum area for filtering.

    Returns:
        List[List[Tuple[int, int, int, int]]], List[List[float]]: per-image list of
        (x_min, y_min, x_max, y_max) and per-image list of confidences.
    """
    B, _, H, W = masks.shape
    masks_np = masks.cpu().numpy().astype(np.uint8)
    conf_np = confidences.cpu().numpy().astype(np.float32)

    all_bbox_results = []
    all_conf_results = []
    all_filtered_masks = []

    for i in range(B):
        mask = masks_np[i, 0]
        conf = conf_np[i, 0]

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        bboxes = []
        conf_results = []
        filtered_mask = []

        for label in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[label]
            if area < minimum_area:
                continue

            component_mask = labels == label
            mean_conf = conf[component_mask].mean() if component_mask.any() else 0.0
            mask = torch.zeros((H, W))
            mask[component_mask] = 1
            filtered_mask.append(mask)
            bboxes.append((x, y, x + w, y + h))
            conf_results.append(float(mean_conf))

        all_bbox_results.append(bboxes)
        all_conf_results.append(conf_results)
        all_filtered_masks.append(filtered_mask)
    return all_bbox_results, all_conf_results, all_filtered_masks

def scale_images(images, new_h, new_w, mode="bilinear", align_corners=False):
    """
    Resizes images from shape (B, nP, 224, 224) to (B, nP, new_h, new_w).

    Args:
        images (torch.Tensor): Input images of shape (B, nP, 224, 224).
        new_h (int): New height.
        new_w (int): New width.
        mode (str): Interpolation mode, e.g., 'bilinear' or 'nearest'.

    Returns:
        torch.Tensor: Resized images of shape (B, nP, new_h, new_w).
    """
    B, nP, H, W = images.shape
    # Combine B and nP into one dimension.
    images_reshaped = images.reshape(B * nP, 1, H, W)

    # Resize the images.
    resized = F.interpolate(images_reshaped, size=(new_h, new_w), mode=mode, align_corners=align_corners)

    # Reshape back to (B, nP, new_h, new_w)
    resized = resized.reshape(B, nP, new_h, new_w)
    return resized

class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

def image_to_tensor(
    img: Union[Image.Image, np.ndarray],
    size: int = 224,
    device: Optional[torch.device] = None,
    normalize: bool = False
) -> torch.Tensor:
    """
    Convert a PIL image or an RGB numpy array into a normalized tensor.

    Args:
        img: 
            - PIL.Image.Image
            - numpy array shape (H, W, 3) in **RGB**
        size: output size (square)
        device: device to move the tensor to
        normalize: apply ImageNet normalization

    Returns:
        Tensor shape (1, 3, size, size)
    """

    # --- Convert numpy array (RGB) → PIL ---
    if isinstance(img, np.ndarray):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"NumPy image must have shape (H, W, 3), but got {img.shape}")
        img = Image.fromarray(img)

    # Make sure it is PIL here
    elif not isinstance(img, Image.Image):
        raise TypeError(f"img must be a PIL.Image or numpy RGB array, got {type(img)}")

    # --- Select normalization ---
    if normalize:
        normalize_trafo = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize_trafo = transforms.Lambda(lambda x: x)

    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        normalize_trafo,
    ])

    # apply transform and add batch dim
    tensor = transform(img).unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor



def center_crop(img: np.ndarray) -> np.ndarray:
    """
    Center-crop an image to a square using the smaller dimension.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C).

    Returns
    -------
    np.ndarray
        The center-cropped square image with side length min(H, W).
    """
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]


def load_image(path: str, *args, **kwargs) -> torch.Tensor:
    """Load image from path and return model-ready tensor (1,3,H,W)."""
    img = Image.open(path).convert("RGB")
    return image_to_tensor(img, *args, **kwargs)


def load_images(paths: Sequence[str],*args, **kwargs) -> torch.Tensor:
    """Load multiple images and return a batched tensor of shape (B,3,H,W).

    This is efficient for small batches and uses the same normalization as image_to_tensor.
    """
    tensors = [load_image(p, *args, **kwargs) for p in paths]
    # each is (1,3,H,W) -> stack into (B,3,H,W)
    return torch.cat(tensors, dim=0)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a CPU tensor to numpy array."""
    return t.detach().cpu().numpy()

def save_nemo(nemo_data: dict[str, Any], out_path: str):
    # Saves nemo_data to a file
    assert "features_3d_updated" in nemo_data.keys(), (
        "No key `features_3d_updated`, make sure you embedded the 3d points into the features!"
    )
    assert "scale_factor" in nemo_data.keys(), (
        "No key `scale_factor`, make sure to set a scale to be able to estimate proper object size!"
    )
    torch.save(nemo_data, out_path)

def load_nemo(path: str, device: Optional[torch.device] = None):
    return torch.load(path, map_location=device)