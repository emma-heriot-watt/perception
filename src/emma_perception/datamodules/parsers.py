import csv
import os

import numpy as np
from tqdm import tqdm


class EpicKitchensParser:
    """Parser for EpicKitchens annotations."""

    def __init__(self, annotation_csv: str, downsample: int = 0) -> None:
        self.metadata = self.parse_annotations(annotation_csv)
        self.downsample = downsample

    def parse_annotations(self, ann_csv: str) -> dict[str, dict[int, int]]:
        """Parsers the annotation csv file into a dictionary.

        Currently storing only start and end times of annotations.
        """
        metadata = {}
        with open(ann_csv) as fp:
            csv_reader = csv.reader(fp)
            header = next(csv_reader)
            indices = list(range(len(header)))
            idx2column = dict(zip(header, indices))
            for row in tqdm(csv_reader, desc="Parsing epic kitchens annotations"):
                video_id = row[idx2column["video_id"]]
                start = int(row[idx2column["start_frame"]])
                end = int(row[idx2column["stop_frame"]])
                if video_id not in metadata:
                    metadata[video_id] = {start: end}
                else:
                    metadata[video_id][start] = end
        return metadata

    def valid(self, video_id: str, frame_file: str) -> bool:
        """Checks if a frame corresponds to an annotation."""
        valid = False
        if video_id in self.metadata.keys():
            start_fr, end_fr = (
                list(self.metadata[video_id].keys()),
                list(self.metadata[video_id].values()),
            )
            bname = os.path.splitext(os.path.basename(frame_file))[0]
            frame_id = int(bname.split("_")[-1])
            start_valid = [start - 1 < frame_id for start in start_fr]
            end_valid = [frame_id < end + 1 for end in end_fr]
            overlap = [start and end for start, end in zip(start_valid, end_valid)]
            valid = any(overlap)

            if self.downsample and valid:
                overlap_indices = np.where(overlap)[0]
                for idx in overlap_indices:
                    start, end = start_fr[idx], end_fr[idx]
                    num_frames = len(np.arange(start, end + 1))
                    tick = float(num_frames / self.downsample)
                    downsampled_frames = [
                        start + int(tick * idx) for idx in range(self.downsample)
                    ]
                    valid = frame_id in downsampled_frames
        return valid

    def get_frames(self, video_path: str) -> list[str]:
        """Return the frames that correspond to annotated segments for a given video."""
        frames = sorted(os.listdir(video_path))
        return [
            os.path.join(video_path, frame)
            for frame in frames
            if self.valid(os.path.basename(video_path), frame)
        ]
