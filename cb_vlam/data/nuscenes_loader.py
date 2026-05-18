"""nuScenes dataset iteration for concept mining.

Wraps the nuscenes-devkit to yield per-sample records containing everything
the extractors need: ego pose, annotations, HD map handle, and front camera
image.
"""

from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import numpy as np


# nuScenes has 4 map locations. We load them lazily.
_VALID_MAP_LOCATIONS = {
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
}


class NuScenesLoader:
    """Iterates nuScenes samples in scene order.

    Yields one record per sample (keyframe at 2Hz). Each record contains
    all data needed for the three concept extraction passes.

    Per AutoVLA convention, ego pose is read from the LIDAR_TOP sample_data
    record — that's the canonical sample-time pose used across nuScenes
    downstream tooling.
    """

    def __init__(self,
                 data_root: Path,
                 version: str = "v1.0-mini",
                 verbose: bool = True,
                 load_images: bool = True):
        """
        Args:
            data_root: Path to the nuScenes data root (containing v1.0-mini/, samples/, etc.).
            version: nuScenes version string. "v1.0-mini" for development, "v1.0-trainval" for full.
            verbose: Pass to NuScenes constructor.
            load_images: If False, skip loading the front-camera image into memory
                (front_image will be None). Path is still populated. Useful for
                pass-A-only runs where the image isn't needed.
        """
        self.data_root = Path(data_root)
        self.version = version
        self.verbose = verbose
        self.load_images = load_images
        self._nusc = None
        self._maps: Dict[str, Any] = {}

    def _load(self) -> None:
        """Lazy-load the nuScenes object on first use."""
        if self._nusc is not None:
            return
        from nuscenes.nuscenes import NuScenes
        self._nusc = NuScenes(
            version=self.version,
            dataroot=str(self.data_root),
            verbose=self.verbose,
        )

    def _get_map(self, location: str) -> Any:
        """Get or load the NuScenesMap for a given location (e.g., 'boston-seaport')."""
        if location not in _VALID_MAP_LOCATIONS:
            raise ValueError(
                f"Unknown nuScenes map location: {location!r}. "
                f"Expected one of {sorted(_VALID_MAP_LOCATIONS)}."
            )
        if location not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            self._maps[location] = NuScenesMap(
                dataroot=str(self.data_root),
                map_name=location,
            )
        return self._maps[location]

    def iter_scenes(self,
                    max_scenes: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate scenes, yielding scene metadata.

        Args:
            max_scenes: Optional limit on number of scenes (for development).

        Yields:
            Dict with keys:
                - "scene_token": str
                - "scene_name": str
                - "location": str  (e.g., "boston-seaport")
                - "nusc_map": NuScenesMap instance
                - "first_sample_token": str
        """
        self._load()
        scenes = self._nusc.scene
        if max_scenes is not None:
            scenes = scenes[:max_scenes]
        for scene in scenes:
            log = self._nusc.get("log", scene["log_token"])
            location = log["location"]
            yield {
                "scene_token": scene["token"],
                "scene_name": scene["name"],
                "location": location,
                "nusc_map": self._get_map(location),
                "first_sample_token": scene["first_sample_token"],
            }

    def iter_samples(self, scene_token: str) -> Iterator[Dict[str, Any]]:
        """Iterate samples within a scene, in time order.

        Yields:
            Dict with keys:
                - "sample_token": str
                - "timestamp": int  (microseconds)
                - "ego_pose": dict  (translation, rotation, timestamp)
                - "annotations": list of sample_annotation records
                - "front_image": np.ndarray (H, W, 3) uint8, or None if load_images=False
                - "front_image_path": Path
        """
        self._load()
        scene = self._nusc.get("scene", scene_token)
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = self._nusc.get("sample", sample_token)

            # LIDAR_TOP's ego_pose is the canonical sample-time pose
            lidar_sd = self._nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_pose = self._nusc.get("ego_pose", lidar_sd["ego_pose_token"])

            # Sample annotations (3D boxes) by token
            annotations = [
                self._nusc.get("sample_annotation", t)
                for t in sample["anns"]
            ]

            # Front camera path + (optionally) image
            cam_sd = self._nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            front_image_path = self.data_root / cam_sd["filename"]
            front_image = None
            if self.load_images:
                from PIL import Image
                with Image.open(front_image_path) as im:
                    front_image = np.asarray(im.convert("RGB"))

            yield {
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "ego_pose": ego_pose,
                "annotations": annotations,
                "front_image": front_image,
                "front_image_path": front_image_path,
            }

            sample_token = sample["next"]
