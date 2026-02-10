import argparse
import os
import cv2
import torch
from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from nemolib.model import Model
from nemolib.visualization import get_point_cloud_image
from nemolib.utils import save_nemo, load_images, center_crop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint.pth",
        type=str,
        help="Checkpoint to test",
    )
    parser.add_argument("--camera-id", default=0, type=int, help="Camera device ID")
    parser.add_argument("--object-name", default=None, type=str, help="Name of the object. Will be used to create the output folder.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    camera_id = args.camera_id
    # Ask the user for object name
    object_name = args.object_name
    if not object_name:
        object_name = input("Enter the object name: ").strip()
    
    # Create output directory
    out_dir = os.path.join("out", object_name)
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Open camera feed
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    captured_images = []
    count = 0
    print("Press ENTER to capture an image, 'q' to finish capturing and generate NeMO.")
    
    while True:
        ret, frame = cap.read()
        frame = center_crop(frame)
        if not ret:
            print("Failed to grab frame")
            break

        # Display the live feed
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Captured: {len(captured_images)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Camera Feed", display_frame)

        key = cv2.waitKey(1)
        if key == 13:  # ENTER key
            # Save the captured image
            img_path = os.path.join(image_dir, f"{count:03d}.jpg")
            cv2.imwrite(img_path, frame)
            captured_images.append(img_path)
            count += 1
            print(f"Captured image {img_path}")
        elif key & 0xFF == ord('q'):
            # Quit capturing
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if not captured_images:
        print("No images captured, exiting.")
        return

    print(f"Total captured images: {len(captured_images)}")
    
    # Load captured images
    images = load_images(captured_images, device=device, normalize=False)
    images = rearrange(images, 't c h w -> 1 t c h w')

    # Load model
    model = Model.from_checkpoint(args.checkpoint, device=device)
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

    # Update features
    nemo["features_3d_updated"] = nemo["features_3d"] + model.point_encoder(nemo["surface_points"])
    nemo['scale_factor'] = None

    print(f"Saving Neural Memory Object to `{out_dir}`")
    save_nemo(nemo, os.path.join(out_dir, "NeMO.pt"))

    print(f"Template images saved in `{image_dir}`")


if __name__ == "__main__":
    main()