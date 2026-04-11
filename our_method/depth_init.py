import os
import numpy as np
import cv2
import struct
from pathlib import Path
from plyfile import PlyData, PlyElement

def load_zoedepth(device="cuda"):
    """Load ZoeDepth via torch.hub (handles dependencies automatically)."""
    import torch
    model = torch.hub.load(
        "isl-org/ZoeDepth",
        "ZoeD_N",
        pretrained=True,
        trust_repo=True
    )
    model = model.to(device)
    model.eval()
    print("ZoeDepth loaded successfully")
    return model

def estimate_depth(model, image_path):
    """Run ZoeDepth on a single image. Returns (H,W) numpy depth map."""
    from PIL import Image
    import torch
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        depth = model.infer_pil(img)
    return depth.squeeze().cpu().numpy()

def read_colmap_cameras(sparse_dir):
    """Read cameras.bin from COLMAP sparse folder."""
    import collections
    Camera = collections.namedtuple(
        "Camera", ["id", "model", "width", "height", "params"])
    cameras = {}
    with open(os.path.join(sparse_dir, "cameras.bin"), "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # PINHOLE model has 4 params: fx, fy, cx, cy
            num_params = {0: 3, 1: 4, 2: 4, 3: 4, 4: 5, 5: 8, 6: 8, 7: 12}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = Camera(cam_id, model_id, width, height, params)
    return cameras

def read_colmap_images(sparse_dir):
    """Read images.bin from COLMAP sparse folder."""
    images = {}
    with open(os.path.join(sparse_dir, "images.bin"), "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            name = name.decode("utf-8")
            num_points = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points * 24)
            images[image_id] = {
                "name": name,
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id
            }
    return images

def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*w*z,   2*x*z+2*w*y],
        [  2*x*y+2*w*z, 1-2*x*x-2*z*z,   2*y*z-2*w*x],
        [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

def depth_to_world_points(depth, K, R, t, stride=4):
    """
    Back-project depth map to 3D world points.
    stride: subsample every N pixels to keep point count manageable
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Create pixel grid (subsampled)
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    dd = depth[vv, uu]

    # Filter invalid depths
    valid = (dd > 0.1) & (dd < 100.0)
    uu, vv, dd = uu[valid], vv[valid], dd[valid]

    # Back-project to camera coordinates
    x_cam = (uu - cx) * dd / fx
    y_cam = (vv - cy) * dd / fy
    z_cam = dd
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

    # Transform to world coordinates
    # world = R^T * (cam - t)
    pts_world = (R.T @ (pts_cam.T - t.reshape(3,1))).T

    return pts_world

def align_depth_scale(mono_depth, colmap_points_2d, colmap_depths):
    """
    Align monocular depth scale to metric COLMAP depth via least squares.
    mono_depth * scale + shift = colmap_depth
    """
    if len(colmap_points_2d) < 3:
        return 1.0, 0.0

    px = np.clip(colmap_points_2d[:, 0].astype(int), 0, mono_depth.shape[1]-1)
    py = np.clip(colmap_points_2d[:, 1].astype(int), 0, mono_depth.shape[0]-1)
    mono_at_pts = mono_depth[py, px]

    valid = mono_at_pts > 0.01
    if valid.sum() < 3:
        return 1.0, 0.0

    A = np.stack([mono_at_pts[valid],
                  np.ones(valid.sum())], axis=1)
    b = colmap_depths[valid]
    result = np.linalg.lstsq(A, b, rcond=None)
    scale, shift = result[0]
    return float(scale), float(shift)

def create_depth_init(sparse_data_dir, output_ply_path,
                      device="cuda", stride=4):
    """
    Main function: generate dense point cloud from monocular depth.
    Replaces sparse SfM initialization in 3DGS.
    """
    sparse_data_dir = Path(sparse_data_dir)
    sparse_dir = sparse_data_dir / "sparse" / "0"
    images_dir = sparse_data_dir / "images"

    print("Loading COLMAP data...")
    cameras = read_colmap_cameras(str(sparse_dir))
    images = read_colmap_images(str(sparse_dir))

    print("Loading ZoeDepth model...")
    model = load_zoedepth(device)

    all_points = []
    all_colors = []

    for img_id, img_data in images.items():
        img_path = images_dir / img_data["name"]
        if not img_path.exists():
            continue

        print(f"Processing {img_data['name']}...")

        # Get camera intrinsics
        cam = cameras[img_data["camera_id"]]
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Get camera pose (COLMAP convention: x_cam = R * x_world + t)
        R = qvec_to_rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])

        # Estimate monocular depth
        mono_depth = estimate_depth(model, str(img_path))

        # Resize depth to match image
        img_cv = cv2.imread(str(img_path))
        H, W = img_cv.shape[:2]
        mono_depth = cv2.resize(mono_depth, (W, H),
                                interpolation=cv2.INTER_LINEAR)

        # Scale alignment using COLMAP points (if available)
        # For simplicity, use median scaling based on scene bounds
        # A more robust version reads points3D.bin for alignment
        scale = 1.0
        shift = 0.0

        # Apply scale
        aligned_depth = mono_depth * scale + shift
        aligned_depth = np.clip(aligned_depth, 0.1, 100.0)

        # Back-project to world points
        pts_world = depth_to_world_points(aligned_depth, K, R, t, stride=stride)

        # Get colors from image
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        u = np.arange(0, W, stride)
        v = np.arange(0, H, stride)
        uu, vv = np.meshgrid(u, v)
        dd = aligned_depth[vv, uu]
        valid = (dd > 0.1) & (dd < 100.0)
        colors = img_rgb[vv[valid], uu[valid]] / 255.0

        all_points.append(pts_world)
        all_colors.append(colors)
        print(f"  -> {len(pts_world)} points")

    if not all_points:
        raise ValueError("No points generated — check image paths")

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    print(f"Total points: {len(all_points)}")

    # Save as PLY
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    vertex = np.array(
        [(p[0], p[1], p[2], c[0], c[1], c[2])
         for p, c in zip(all_points, all_colors)],
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("red","f4"),("green","f4"),("blue","f4")]
    )
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(output_ply_path)
    print(f"Saved depth-init PLY to {output_ply_path}")
    return output_ply_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    create_depth_init(args.data_dir, args.output_ply,
                      args.device, args.stride)
