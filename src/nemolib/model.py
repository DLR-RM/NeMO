from typing import Any, Dict, Literal, Optional, Union

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch.autograd import Variable
from torchvision import transforms

from .modules.backbone import BackboneOutBlock, build_backbone
from .modules.encoder import CrossViewEncoder
from .modules.lifting import (
    lifting,
    lifting_make_cross_attention_layers,
    lifting_make_decoder_layers,
    lifting_make_self_attention_layers,
)
from .modules.neural_field import NeuralField

from .modules.heads import head_factory


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


class Model(nn.Module):
    def __init__(self, config: Union[DictConfig, ListConfig]):
        super().__init__()
        self.config = config

        # input and output size
        self.input_size = config.dataset.img_size

        # build backbone image encoder
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name
        self.backbone_out_dim = config.model.backbone_out_dim
        self.backbone_out = BackboneOutBlock(
            in_dim=self.backbone_dim, out_dim=self.backbone_out_dim
        )
        self.feat_res = int(self.input_size // self.down_rate)

        # Create a encoder for 3d points that maps it to the same dimension as the backbone
        pos_enc_length = (
            self.config.model.pos_enc_length
            if self.config.model.pos_enc_length > 0
            else None
        )
        self.point_encoder = NeuralField(
            input_dimension=3,
            output_dimension=self.backbone_out_dim,
            hidden_layers=8,
            neurons_per_layer=128,
            pos_enc_length=pos_enc_length,
            output_function=lambda x: x,
        )

        # Create Neural Memory Object
        self.nemo = NeMO(
            in_dim=self.backbone_out_dim,
            in_res=self.feat_res,
            point_encoder=self.point_encoder,
            cross_view_encoder_layers=config.model.encoder_layers,
            lifting_decoder_layers=config.model.lifting_decoder_layers,
            lifting_cross_attention_layers=config.model.lifting_cross_attention_layers,
            lifting_self_attention_layers=config.model.lifting_self_attention_layers,
            norm_first=config.model.norm_first,
            use_flash_attn=config.model.use_flash_attn,
        )

        # Create DenseXZY Predicter
        self.dense_xyz_mapping = DenseXYZPredictor(
            decoder_layers=config.model.decoder_layers,
            cross_attention_layers=config.model.cross_attention_layers,
            self_attention_layers=config.model.self_attention_layers,
            head_config=config,
            head_type=config.model.head_type,
            norm_first=config.model.norm_first,
            use_flash_attn=config.model.use_flash_attn,
        )

        self.img_transformations = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                MaybeToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    @classmethod
    def from_checkpoint(
        cls, ckpt: Union[str, Dict[str, Any]], device: Optional[torch.device] = None
    ) -> "Model":
        """Create encoder from checkpoint. Falls back to a minimal model.

        The returned encoder expects to be used for inference (eval mode).
        """
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt, weights_only=False)
        initial_state_dict = ckpt["state_dict"]
        # Remove `model.` prefix from state_dict
        state_dict = {}
        for key, value in initial_state_dict.items():
            if not key.startswith("model."):
                raise KeyError(
                    "Expecting checkpoint with state_dict containing `model.` prefix!"
                )
            state_dict[key[len("model.") :]] = value
        config = ckpt["hyper_parameters"]["config"]

        model = cls(config)
        if device is None:
            device = torch.device("cpu")
        model.to(device)

        # Attempt to load the state dict if keys match; ignore mismatches.
        model.load_state_dict(state_dict, strict=False)
        print("DONE LOADING CHECKPOINT")
        return model

    def encode_images(
        self,
        images: Union[torch.Tensor, np.ndarray],
        sample_points: Union[torch.Tensor, int] = 1000,
        **kwargs,
    ):
        """
        Encode images into a neural object model.

        Args:
            images: Tensor-like input of shape [b, t, C, H, W].
                    Accepts torch.Tensor or np.ndarray.
            sample_points: Integer for random sampling, or tensor of shape [b, n, 3].
        """

        # --------------------------
        # Normalize images input
        # --------------------------
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                f"'images' must be a torch.Tensor or numpy array, but got {type(images)}"
            )

        if images.ndim != 5:
            raise ValueError(
                f"'images' must have shape [b, t, C, H, W], but got shape {tuple(images.shape)}"
            )

        images = images.float()

        # --------------------------
        # Validate sample_points
        # --------------------------
        if isinstance(sample_points, torch.Tensor):
            if sample_points.ndim != 3 or sample_points.shape[-1] != 3:
                raise ValueError(
                    "'sample_points' tensor must have shape [b, n, 3], "
                    f"but got shape {tuple(sample_points.shape)}"
                )

        elif not isinstance(sample_points, int):
            raise TypeError(
                "'sample_points' must be an integer or a torch.Tensor with shape [b, n, 3], "
                f"but got {type(sample_points)}"
            )

        # --------------------------
        # Call internal model generation
        # --------------------------
        return self._generate_neural_object_model(images, sample_points, **kwargs)

    def decode_images(self, images: Union[torch.Tensor, np.ndarray], features_3d):
        """
        Encode images into a neural object model.

        Args:
            images: Tensor-like input of shape [b, t, C, H, W].
                    Accepts torch.Tensor or np.ndarray.
            sample_points: Integer for random sampling, or tensor of shape [b, n, 3].
        """

        # --------------------------
        # Normalize images input
        # --------------------------
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                f"'images' must be a torch.Tensor or numpy array, but got {type(images)}"
            )

        if images.ndim != 5:
            raise ValueError(
                f"'images' must have shape [b, t, C, H, W], but got shape {tuple(images.shape)}"
            )

        images = images.float()

        return self._predict_xyz_mapping(images, features_3d)

    def forward(self, run: Literal["nemo", "xyz"], *args, **kwargs):
        if run == "nemo":
            return self._generate_neural_object_model(*args, **kwargs)
        elif run == "xyz":
            return self._predict_xyz_mapping(*args, **kwargs)
        elif run == "pnp_pose_estimation":
            return self._pnp_pose_estimation(*args, **kwargs)
        else:
            raise AttributeError(f"Unknown run type {run=}")

    def extract_feature(self, x):
        if self.backbone_name == "dinov2":
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = (
                int(h_origin / self.backbone.patch_embed.patch_size[0]),
                int(w_origin / self.backbone.patch_embed.patch_size[1]),
            )
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError("unknown image backbone")
        return out

    def _generate_neural_object_model(
        self, images, sample_points: Union[torch.Tensor, int] = 1000, **kwargs
    ):
        """
        imgs in shape [b,t,C,H,W]
        sample_points in shape [b, n, 3] or integer, which means random sampling
        """
        b, t, _, _, _ = images.shape
        imgs = self.img_transformations(rearrange(images, "b t c h w -> (b t) c h w"))

        # 2D per-view feature extraction
        if self.config.model.backbone_fix:
            with torch.no_grad():
                features = self.extract_feature(imgs)  # [b*t,c=768,h,w]
        else:
            features = self.extract_feature(imgs)
        features = self.backbone_out(features)
        features = rearrange(
            features, "(b t) c h w -> b t c h w", b=b, t=t
        )  # [b,t,c,h,w]

        # Create Neural Object Model
        result = self.nemo(features, sample_points, **kwargs)

        return result

    def _predict_xyz_mapping(self, images, features_3d):
        b, t, _, _, _ = images.shape
        imgs = self.img_transformations(rearrange(images, "b t c h w -> (b t) c h w"))
        _, _, h, w = imgs.shape  # Take height and width from transformed images

        # 2D per-view feature extraction
        if self.config.model.backbone_fix:
            with torch.no_grad():
                img_features = self.extract_feature(imgs)  # [b*t,c=768,h,w]
        else:
            img_features = self.extract_feature(imgs)
        img_features = self.backbone_out(img_features)
        img_features = rearrange(
            img_features, "(b t) c h w -> b t c h w", b=b, t=t
        )  # [b,t,c,h,w]

        result = self.dense_xyz_mapping.forward(
            img_features, features_3d, img_shape=(h, w)
        )
        return result

    def _pnp_pose_estimation(
        self,
        xyz_results,
        scene_bboxes,
        intrinsic,
        conf_threshold: float = 0.1,
        nemo_scale_factor: float = 1.0,
        **kwargs,
    ):
        pts3d = xyz_results["pts3d"]
        conf = xyz_results["conf"]

        b, t, h, w, c = pts3d.shape
        b, t, h, w = conf.shape

        new_xmin, new_ymin, new_xmax, new_ymax = (
            scene_bboxes[..., 0].cpu().numpy(),
            scene_bboxes[..., 1].cpu().numpy(),
            scene_bboxes[..., 2].cpu().numpy(),
            scene_bboxes[..., 3].cpu().numpy(),
        )
        new_width, new_height = new_xmax - new_xmin, new_ymax - new_ymin
        assert (new_width == new_height).all(), "Bounding Box is not rectangular!"
        bbox_length = new_width
        bbox_scale = w / bbox_length
        # We have to scale the position of the pixels as if the original image would have created a crop of 224x224
        # Therefore we use the bbox scale
        conf_2d_3d = rearrange(conf.cpu().numpy(), "b t h w -> (b t) (h w)")
        indices = np.repeat(np.indices((224, 224))[np.newaxis, ...], b * t, axis=0)
        points_2d = rearrange(indices, "bt c h w -> bt (h w) c").astype(np.float32)
        # Swap columns to get (x, y) order:
        points_2d = points_2d[..., [1, 0]]
        points_2d[..., 0] += new_xmin * bbox_scale
        points_2d[..., 1] += new_ymin * bbox_scale
        points_3d = (
            rearrange(pts3d, "b t h w c -> (b t) (h w) c")
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        points_3d *= float(nemo_scale_factor)
        # Scale camera intrinsic
        intrinsic = rearrange(intrinsic, "b t h w -> (b t) h w")
        K_scaled = intrinsic * bbox_scale.squeeze()[..., None, None]
        K_scaled[..., 2, 2] = 1
        # Apply conf threshold
        valid = conf_2d_3d > conf_threshold

        estimated_poses = _run_pnp_on_multiple_images(
            valid, points_2d, points_3d, K_scaled, **kwargs
        )

        return estimated_poses


def _run_pnp_on_multiple_images(
    valid,
    points_2d,
    points_3d,
    K_scaled,
    confidence=0.99999,
    iterationsCount=100_000,
    reprojectionError=5,
    min_inlier_ratio=0.0,
    verbose: bool = False,
):
    estTs = []  # To store the transformation matrices for each image
    B = len(valid)  # Number of images in the batch
    for b in range(B):
        # Apply confidence threshold
        valid_points = valid[b]  # Extract valid mask for current image
        valid_2d = points_2d[b][valid_points]  # Points in 2D for current image
        valid_3d = points_3d[b][valid_points]  # Points in 3D for current image

        if len(valid_3d) < 16 * 16:
            print(f"Not enough valid points for image {b}, skipping PnP")
            estTs.append(None)
            continue

        # Camera matrix for current image
        K = K_scaled[b]

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                valid_3d,
                valid_2d,
                K,
                None,
                flags=cv2.SOLVEPNP_SQPNP,
                iterationsCount=iterationsCount,
                reprojectionError=reprojectionError,
                confidence=confidence,
            )
        except Exception as e:
            print(f"Error in PnP for image {b}: {e}")
            success = False
        if success and inliers is not None:
            inlier_ratio = len(inliers) / len(valid_2d)
            if verbose:
                print(f"PnP Inlier Ratio: {inlier_ratio:.2%}")
            if inlier_ratio < min_inlier_ratio:
                success = False
        if not success:
            print(f"Failed to run PnP for image {b}")
            estTs.append(None)
            continue

        # Construct 4x4 homogeneous transformation matrix
        estT = np.eye(4)  # Start with identity matrix
        estT[:3, :3] = cv2.Rodrigues(rvec)[0]  # Set rotation part
        estT[:3, 3] = tvec.flatten()  # Set translation part

        # Append the transformation matrix for the current image
        estTs.append(estT)

    return estTs


class NeMO(nn.Module):
    def __init__(
        self,
        in_dim: int,
        in_res: int,
        point_encoder: nn.Module,
        cross_view_encoder_layers: int,
        lifting_decoder_layers: int,
        lifting_cross_attention_layers: int,
        lifting_self_attention_layers: int,
        norm_first: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()

        # build cross-view feature encoder that attents between images
        self.cross_view_encoder = CrossViewEncoder(
            in_dim=in_dim,
            in_res=in_res,
            layers=cross_view_encoder_layers,
            norm_first=norm_first,
            use_flash_attn=use_flash_attn,
        )

        self.point_encoder = point_encoder

        # build 2D-3D lifting
        self.lifting = lifting(
            in_dim,
            lifting_decoder_layers,
            lifting_cross_attention_layers,
            lifting_self_attention_layers,
            norm_first=norm_first,
            use_flash_attn=use_flash_attn,
        )

        # build Neural Unsigned Distance Field
        self.nudf = NeuralField(
            hidden_layers=16,
            neurons_per_layer=512,
            input_dimension=in_dim,
            output_dimension=1,
            output_function=lambda x: 2 * (1 + torch.tanh(x)),
        )

    def forward(
        self,
        img_features: torch.Tensor,
        sample_points: Union[torch.Tensor, int] = 1000,
    ):
        """
        img_features: features in shape [b,t,C,H,W]
        sample_points: points in shape [b, n, 3] or integer, which means random sampling
        """
        grad_enabled = torch.is_grad_enabled()
        b = img_features.shape[0]
        device = img_features.device

        # Sample points are 3D points form -1 to 1
        if isinstance(sample_points, int):
            sample_points = torch.rand(b, sample_points, 3) * 2 - 1  # [b,n,3]
        elif isinstance(sample_points, torch.Tensor):
            sample_points = sample_points
        sample_points = sample_points.to(device)

        assert (
            sample_points.shape[0] == b
        ), "Sample points has to have the same batch size as input images"

        with torch.set_grad_enabled(True):
            sample_points_backprop = Variable(sample_points, requires_grad=True)

            # cross-view feature refinement
            features = self.cross_view_encoder(img_features)  # [b,t,c,h,w]

            # point embedding
            features_points3d = self.point_encoder(sample_points_backprop)

            # 2D-3D lifting
            features_3d = self.lifting(features, features_points3d)  # [b,n,c]

            # unsigned distance prediction
            distances = self.nudf(features_3d)  # [b,n,1]

            # Calculate surface point based on input sample points, gradient and distance
            (gradients,) = torch.autograd.grad(
                distances,
                sample_points_backprop,
                grad_outputs=torch.ones_like(distances),
                retain_graph=grad_enabled,
                create_graph=True,
            )  # [b, n, 3]

        gradients = torch.nn.functional.normalize(gradients, dim=2)  # [b,n,3]
        surface_points = sample_points_backprop - distances * gradients  # [b,n,3]
        features_3d = features_3d

        if not grad_enabled:
            distances = distances.detach()
            gradients = gradients.detach()
            surface_points = surface_points.detach()
            sample_points = sample_points.detach()
            features_3d = features_3d.detach()

        results = {}
        results["distances"] = distances  # [b, n, 1]
        results["gradients"] = gradients  # [b, n, 3]
        results["surface_points"] = surface_points  # [b, n, 3]
        results["sample_points"] = sample_points  # [b, n, 4]
        results["features_3d"] = features_3d  # [b, n, c]

        return results


class DenseXYZPredictor(nn.Module):
    def __init__(
        self,
        decoder_layers: int,
        cross_attention_layers: int,
        self_attention_layers: int,
        head_config: Union[DictConfig, ListConfig],
        head_type: Literal["dpt", "linear"] = "dpt",
        norm_first: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        # Build image <-> geometry attention
        ## Create decoder layers
        self.decoder = lifting_make_decoder_layers(
            in_dim=768,
            layers=decoder_layers,
            norm_first=norm_first,
            use_flash_attn=use_flash_attn,
        )

        ## Create a cross attention module (q is image tokens, v is 3d point embeddings from nudf)
        self.cross_attention = lifting_make_cross_attention_layers(
            in_dim=768,
            layers=cross_attention_layers,
            norm_first=norm_first,
            use_flash_attn=use_flash_attn,
        )
        ## Self attention
        self.self_attention = lifting_make_self_attention_layers(
            in_dim=768,
            layers=self_attention_layers,
            norm_first=norm_first,
            use_flash_attn=use_flash_attn,
        )

        # Build regression head for point map and confidence map
        head_type = head_type  # dpt, linear
        output_mode = "pts3d"
        self.regression_head = head_factory(head_type, output_mode, head_config)

    def forward(
        self,
        img_features: torch.Tensor,
        features_3d: torch.Tensor,
        img_shape: tuple[int, int],
    ):
        img_features = img_features
        features_3d = features_3d
        h, w = img_shape
        b, t, fc, fh, fw = img_features.shape
        features_2d = rearrange(img_features, "b t c h w -> (b t) (h w) c")

        features_3d_updated = einops.repeat(
            features_3d, "b n c -> (b copy) n c", copy=t
        )

        features_2d_list = [features_2d]
        # DECODER
        for block in self.decoder:
            features_2d_list.append(block(features_2d_list[-1], features_3d_updated))

        # SA+CA
        for ca, sa in zip(self.cross_attention, self.self_attention):
            features_2d_list.append(ca(features_2d_list[-1], features_3d_updated))
            features_2d_list.append(sa(features_2d_list[-1]))

        features_2d_list = [
            rearrange(f, "b (h w) c -> b c h w", h=fh, w=fw) for f in features_2d_list
        ]

        # Predict depth based on updated 2d features
        res = self.regression_head(
            [tok.float() for tok in features_2d_list[1:]], (h, w)
        )
        # res has entry 'pts3d' and 'conf'
        res["pts3d"] = rearrange(res["pts3d"], "(b t) h w c -> b t h w c", b=b, t=t)
        res["conf"] = rearrange(res["conf"], "(b t) h w -> b t h w", b=b, t=t)
        res["depth_scaled"] = rearrange(
            res["depth_scaled"], "(b t) h w -> b t h w", b=b, t=t
        )
        res["depth_scaled_conf"] = rearrange(
            res["depth_scaled_conf"], "(b t) h w -> b t h w", b=b, t=t
        )
        res["mask"] = rearrange(res["mask"], "(b t) h w -> b t h w", b=b, t=t)
        res["mask_full"] = rearrange(res["mask_full"], "(b t) h w -> b t h w", b=b, t=t)

        return res
