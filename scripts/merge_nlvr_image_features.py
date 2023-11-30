import argparse
import glob
from pathlib import Path, PurePath

import torch
from tqdm import tqdm


def merge_nlvr_image_features(input_path: Path, output_path: Path) -> None:
    """Merge NLVR image features into video features."""
    output_path.mkdir(exist_ok=True)

    img_left_paths = sorted(glob.glob(str(input_path.joinpath("*img0.pt"))))
    img_right_paths = sorted(glob.glob(str(input_path.joinpath("*img1.pt"))))
    if len(img_left_paths) != len(img_right_paths):
        raise AssertionError(
            f"The number of 0-indexed images {len(img_left_paths)} should match the number of 1-indexed images {len(img_right_paths)}"
        )

    img_file_paths = list(zip(img_left_paths, img_right_paths))
    for img_left_path, img_right_path in tqdm(img_file_paths):
        if img_left_path.split("img0")[0] != img_right_path.split("img1")[0]:
            raise AssertionError(f"Found unpaired images {img_left_path} {img_right_path}")
        img_left_features = torch.load(img_left_path)
        img_right_features = torch.load(img_right_path)

        features_dict = {
            "frames": [
                {"image": PurePath(img_left_path).name, "features": img_left_features},
                {"image": PurePath(img_right_path).name, "features": img_right_features},
            ]
        }

        output_file_path = output_path.joinpath(
            PurePath(f"{img_left_path.split('img0')[0][:-1]}.pt").name
        )
        torch.save(features_dict, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=Path,
        help="Path to input nlvr image features",
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to input nlvr image features",
        required=True,
    )

    args = parser.parse_args()
    merge_nlvr_image_features(input_path=args.input_path, output_path=args.output_path)
