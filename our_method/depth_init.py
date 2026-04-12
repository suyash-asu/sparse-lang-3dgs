import os
import numpy as np
import cv2
import struct
from pathlib import Path
from plyfile import PlyData, PlyElement


def load_depth_model(device="cuda"):
    """Load MiDaS DPT-Large for monocular depth estimation."""
    import torch
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    model = model.to(device)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.dpt_transform
    print("MiDaS DPT-Large loaded successfully")
    return model, transform


def estimate_depth(model, transform, image_path, device="cuda"):
    """Run MiDaS on a single image. Returns (H,W) numpy depth map."""
    import torch
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = 1.0 - depth
    return depth


def read_colmap_cameras(sparse_dir):
    """Read cameras.bin using 3DGS's proven COLMAP reader."""
    import sys
    import collections
    gs_path = str(Path(__file__).parent.parent.parent / "gaussian-splatting")
    if gs_path not in sys.path:
        sys.path.insert(0, gs_path)
    from scene.colmap_loader import read_intrinsics_binary
    Camera = collections.namedtuple(
        "Camera", ["id", "model", "width", "height", "params"])
    cam_intrinsics = read_intrinsics_binary(
        os.path.join(sparse_dir, "cameras.bin"))
    return {cam_id: Camera(i.id, i.model, i.width, i.height, i.params)
            for cam_id, i in cam_intrinsics.items()}

def read_colmap_points3d(sparse_dir):
    """Read points3D.bin — returns dict of point_id -> xyz (world coords)."""
    points3d = {}
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    with open(bin_path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            pt_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)   # rgb (3 bytes)
            f.read(8)   # error (double)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # track: image_id (int32) + point2D_idx (int32)
            points3d[pt_id] = np.array(xyz)
    return points3d

def read_colmap_images(sparse_dir):
    """Read images.bin using 3DGS's proven COLMAP reader."""
    import sys
    gs_path = str(Path(__file__).parent.parent.parent / "gaussian-splatting")
    if gs_path not in sys.path:
        sys.path.insert(0, gs_path)
    from scene.colmap_loader import read_extrinsics_binary
    cam_extrinsics = read_extrinsics_binary(
        os.path.join(sparse_dir, "images.bin"))
    images = {}
    for img_id, extr in cam_extrinsics.items():
        images[img_id] = {
            "name": extr.name,
            "qvec": np.array(extr.qvec),
            "tvec": np.array(extr.tvec),
            "camera_id": extr.camera_id,
            "xys": np.array(extr.xys),               # (N, 2)
            "point3D_ids": np.array(extr.point3D_ids) # (N,)
        }
    return images


def qvec_to_rotmat(qvec):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1-2*y*y-2*z*z,  2*x*y-2*w*z,  2*x*z+2*w*y],
        [  2*x*y+2*w*z,1-2*x*x-2*z*z,  2*y*z-2*w*x],
        [  2*x*z-2*w*y,  2*y*z+2*w*x,1-2*x*x-2*y*y]
    ])


def align_scale_to_colmap(mono_depth, img_data, points3d, K, R, t):
    """
    Align MiDaS relative depth to metric COLMAP scale using visible
    3D points as anchors. Solves: mono * scale + shift = metric_depth
    """
    H, W = mono_depth.shape
    xys = img_data["xys"]            # (M, 2)
    pt_ids = img_data["point3D_ids"] # (M,)

    sfm_mono = []
    sfm_metric = []

    for i in range(len(pt_ids)):
        pt_id = int(pt_ids[i])
        if pt_id == -1 or pt_id not in points3d:
            continue

        x2d, y2d = xys[i]
        px = int(np.clip(x2d, 0, W - 1))
        py = int(np.clip(y2d, 0, H - 1))
        mono_val = mono_depth[py, px]
        if mono_val < 0.01:
            continue

        # Project 3D point to get metric depth
        pt_cam = R @ points3d[pt_id] + t
        depth_metric = pt_cam[2]
        if depth_metric < 0.1:
            continue

        sfm_mono.append(mono_val)
        sfm_metric.append(depth_metric)

    n = len(sfm_mono)
    if n < 3:
        print(f"  Warning: only {n} anchor points — using scale=1")
        return 1.0, 0.0

    A = np.stack([sfm_mono, np.ones(n)], axis=1)
    result = np.linalg.lstsq(A, np.array(sfm_metric), rcond=None)
    scale, shift = result[0]
    print(f"  Scale alignment: scale={scale:.4f}, shift={shift:.4f} "
          f"from {n} anchors")
    return float(scale), float(shift)


def depth_to_world_points(depth, K, R, t, stride=4):
    """Back-project depth map to 3D world points."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    dd = depth[vv, uu]

    valid = (dd > 0.1) & (dd < 100.0)
    uu, vv, dd = uu[valid], vv[valid], dd[valid]

    x_cam = (uu - cx) * dd / fx
    y_cam = (vv - cy) * dd / fy
    z_cam = dd
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # COLMAP: x_cam = R @ x_world + t  =>  x_world = R^T @ (x_cam - t)
    pts_world = (R.T @ (pts_cam.T - t.reshape(3, 1))).T
    return pts_world


def create_depth_init(sparse_data_dir, output_ply_path,
                      original_data_dir=None,
                      device="cuda", stride=4):
    """
    Generate a dense depth-lifted point cloud from monocular depth estimates.
    Depth is scale-aligned to COLMAP metric coordinates using sparse 3D
    points as anchors. Output PLY replaces points3D.ply for 3DGS init.
    """
    sparse_data_dir = Path(sparse_data_dir)
    sparse_dir = sparse_data_dir / "sparse" / "0"
    images_dir = sparse_data_dir / "images"

    print("Loading COLMAP data...")
    cameras = read_colmap_cameras(str(sparse_dir))
    images = read_colmap_images(str(sparse_dir))

    # ── NEW BLOCK: load tracks and points from original full dataset ──
    if original_data_dir is not None:
        orig_sparse = Path(original_data_dir) / "sparse" / "0"
        print("Loading full-dataset images for scale alignment anchors...")
        images_with_tracks = read_colmap_images(str(orig_sparse))
        points3d = read_colmap_points3d(str(orig_sparse))
        print(f"  {len(points3d)} 3D anchor points loaded")
        # Copy 2D track data into our subsampled images by matching name
        name_to_orig = {od["name"]: od for od in images_with_tracks.values()}
        for img_id, img_data in images.items():
            name = img_data["name"]
            if name in name_to_orig:
                orig = name_to_orig[name]
                img_data["xys"] = orig["xys"]
                img_data["point3D_ids"] = orig["point3D_ids"]  # ← this was missing
                print(f"  Copied {np.sum(orig['point3D_ids'] != -1)} "
                      f"tracks for {name}")
    else:
        print("Loading COLMAP 3D points for scale alignment...")
        points3d = read_colmap_points3d(str(sparse_dir))
        print(f"  {len(points3d)} 3D anchor points loaded")

    print("Loading MiDaS depth model...")
    model, transform = load_depth_model(device)

    all_points = []
    all_colors = []

    for img_id, img_data in images.items():
        img_path = images_dir / img_data["name"]
        if not img_path.exists():
            continue

        print(f"Processing {img_data['name']}...")

        # Camera intrinsics
        cam = cameras[img_data["camera_id"]]
        fx = cam.params[0]
        fy = cam.params[1] if len(cam.params) > 1 else cam.params[0]
        cx = cam.params[2] if len(cam.params) > 2 else cam.width / 2
        cy = cam.params[3] if len(cam.params) > 3 else cam.height / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Camera pose
        R = qvec_to_rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])

        # Estimate relative depth with MiDaS
        mono_depth = estimate_depth(model, transform, str(img_path), device)

        # Resize depth to match original image size
        img_cv = cv2.imread(str(img_path))
        H, W = img_cv.shape[:2]
        mono_depth = cv2.resize(mono_depth, (W, H),
                                interpolation=cv2.INTER_LINEAR)

        # Align MiDaS relative depth to COLMAP metric scale
        scale, shift = align_scale_to_colmap(
            mono_depth, img_data, points3d, K, R, t)

        # Apply alignment
        aligned_depth = mono_depth * scale + shift
        aligned_depth = np.clip(aligned_depth, 0.1, 100.0)

        # Back-project to world coordinates
        pts_world = depth_to_world_points(aligned_depth, K, R, t,
                                          stride=stride)

        # Get colors from RGB image
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

    # Save PLY with normals (required by 3DGS)
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    colors_uint8 = (all_colors * 255).clip(0, 255).astype(np.uint8)
    normals = np.zeros_like(all_points)

    vertex = np.array(
        [(p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2])
         for p, n, c in zip(all_points, normals, colors_uint8)],
        dtype=[("x","f4"),("y","f4"),("z","f4"),
               ("nx","f4"),("ny","f4"),("nz","f4"),
               ("red","u1"),("green","u1"),("blue","u1")]
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
    parser.add_argument("--original_data_dir", default=None,
                    help="Original full dataset dir for scale alignment")
    args = parser.parse_args()
    create_depth_init(args.data_dir, args.output_ply,
                  original_data_dir=args.original_data_dir,
                  device=args.device,
                  stride=args.stride)
