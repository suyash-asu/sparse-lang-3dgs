import os
import shutil
import numpy as np
import argparse
import struct

def read_images_binary(path):
    """Read COLMAP images.bin and return dict of image_id -> image_name"""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_points = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points * 24)
            images[image_id] = {
                "name": name,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "xys": [],
                "point3D_ids": []
            }
    return images

def write_images_binary(images, path):
    """Write filtered images back to binary format"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, img in images.items():
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", *img["qvec"]))
            f.write(struct.pack("<3d", *img["tvec"]))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))

def subsample_views(data_dir, output_dir, n_views, seed=42):
    np.random.seed(seed)

    images_dir = os.path.join(data_dir, "images")
    all_images = sorted(os.listdir(images_dir))
    total = len(all_images)

    indices = np.linspace(0, total - 1, n_views, dtype=int)
    selected = set(all_images[i] for i in indices)

    print(f"Total images: {total}")
    print(f"Selected {n_views} views: {sorted(selected)}")

    # Copy selected images
    out_images = os.path.join(output_dir, "images")
    os.makedirs(out_images, exist_ok=True)
    for img in selected:
        shutil.copy(
            os.path.join(images_dir, img),
            os.path.join(out_images, img)
        )

    # Copy sparse folder
    in_sparse = os.path.join(data_dir, "sparse", "0")
    out_sparse = os.path.join(output_dir, "sparse", "0")
    os.makedirs(out_sparse, exist_ok=True)

    # Copy cameras.bin and points3D.bin as-is
    shutil.copy(os.path.join(in_sparse, "cameras.bin"),
                os.path.join(out_sparse, "cameras.bin"))
    shutil.copy(os.path.join(in_sparse, "points3D.bin"),
                os.path.join(out_sparse, "points3D.bin"))

    # Filter images.bin to only selected images
    all_cam_data = read_images_binary(
        os.path.join(in_sparse, "images.bin"))
    filtered = {k: v for k, v in all_cam_data.items()
                if v["name"] in selected}
    write_images_binary(filtered,
                        os.path.join(out_sparse, "images.bin"))

    print(f"Filtered COLMAP to {len(filtered)} cameras")
    print(f"Saved to {output_dir}")
    return sorted(selected)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_views", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subsample_views(args.data_dir, args.output_dir,
                    args.n_views, args.seed)
