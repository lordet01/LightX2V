import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
    HAND_CONNECTIONS = list(_mp_hands.HAND_CONNECTIONS)
except Exception:
    _mp_hands = None
    HAND_CONNECTIONS = []


TECH_DEMO_PATH = "/works/z-profile/tech_demo"
if TECH_DEMO_PATH not in sys.path:
    sys.path.append(TECH_DEMO_PATH)

from t1_landmark import T1LandmarkDetector  # noqa: E402
from t3_warping import (  # noqa: E402
    PLASTIC_SURGERY_LINES,
    compute_robust_midline,
    symmetrize_landmarks,
)


Point = Tuple[float, float]
IndexedPointMap = Dict[int, Point]
Point3D = Tuple[float, float, float]
IndexedPoint3DMap = Dict[int, Point3D]


POSE_LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6),  # eyes
    (7, 8),  # ears
    (9, 10),  # mouth corners
    (11, 12), (13, 14), (15, 16),  # arms
    (17, 18), (19, 20), (21, 22),  # hands
    (23, 24), (25, 26), (27, 28),  # legs
    (29, 30), (31, 32),  # feet
]

POSE_CENTER_INDICES = [0]  # nose
POSE_FACE_SYNC_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
POSE_DRAW_CONNECTIONS = [
    # face/head
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (30, 32),
]


def _extract_indices_from_structure(data) -> Set[int]:
    indices: Set[int] = set()
    if isinstance(data, int):
        indices.add(data)
    elif isinstance(data, list):
        for item in data:
            indices.update(_extract_indices_from_structure(item))
    elif isinstance(data, dict):
        for value in data.values():
            indices.update(_extract_indices_from_structure(value))
    return indices


MAJOR_FACE_LANDMARK_INDICES = _extract_indices_from_structure(PLASTIC_SURGERY_LINES)


@dataclass
class StressTestOptions:
    enabled: bool = False
    face_reverse_strength: float = 0.0
    face_roll_deg: float = 0.0
    face_shear: float = 0.0
    face_x_shift_ratio: float = 0.0
    face_y_shift_ratio: float = 0.0
    left_arm_angle_deg: Optional[float] = None
    right_arm_angle_deg: Optional[float] = None
    arm_extension: float = 1.0
    hand_spread_px: float = 24.0
    sync_pose_face_with_eye_mouth: bool = True
    force_pose_face_keypoint_match: bool = True
    auto_to_frontal: bool = True
    target_yaw_deg: float = 0.0
    target_pitch_deg: float = 0.0
    target_roll_deg: float = 0.0
    camera_focal_scale: float = 1.2
    camera_base_depth_scale: float = 1.6
    face_depth_scale: float = 1.0
    pose_depth_scale: float = 1.0
    use_torso_plane_facing: bool = True
    torso_facing_strength: float = 0.6
    min_depth_px: float = 60.0
    max_reprojection_error_px: float = 400.0
    apply_frontal_affine: bool = True
    frontal_center_x_ratio: float = 0.5
    frontal_eye_y_ratio: float = 0.42
    frontal_mouth_y_ratio: float = 0.60
    frontal_eye_half_dist_ratio: float = 0.13


def make_stress_test_options(profile: str = "none") -> StressTestOptions:
    if profile == "strong_reverse":
        return StressTestOptions(
            enabled=True,
            face_reverse_strength=1.0,
            face_roll_deg=-28.0,
            face_shear=-0.30,
            face_x_shift_ratio=-0.08,
            face_y_shift_ratio=-0.02,
            left_arm_angle_deg=-155.0,
            right_arm_angle_deg=20.0,
            arm_extension=1.35,
            hand_spread_px=42.0,
        )
    if profile == "pose_only_extreme":
        return StressTestOptions(
            enabled=True,
            left_arm_angle_deg=-170.0,
            right_arm_angle_deg=35.0,
            arm_extension=1.45,
            hand_spread_px=44.0,
        )
    return StressTestOptions()


def _clamp_to_int(v: float, min_v: int, max_v: int) -> int:
    return max(min_v, min(int(round(v)), max_v))


def _is_valid_pose_point_for_overlay(
    x: float,
    y: float,
    width: int,
    height: int,
    edge_margin: int = 1,
) -> bool:
    """
    Exclude points on/near image boundary (typically clipped stress-test artifacts).
    """
    return (
        edge_margin <= x < (width - edge_margin)
        and edge_margin <= y < (height - edge_margin)
    )


def _get_pose_color(idx: int) -> Tuple[int, int, int]:
    if idx >= 1000:
        return (80, 255, 80)  # virtual torso guides: bright green
    if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        return (0, 255, 255)  # head/face: cyan
    if idx in [11, 12, 23, 24]:
        return (0, 255, 0)  # torso anchors: green
    if idx in [13, 15, 17, 19, 21]:
        return (255, 180, 0)  # left arm: orange
    if idx in [14, 16, 18, 20, 22]:
        return (60, 120, 255)  # right arm: blue
    if idx in [25, 27, 29, 31]:
        return (255, 0, 255)  # left leg: magenta
    if idx in [26, 28, 30, 32]:
        return (180, 0, 255)  # right leg: purple
    return (255, 255, 0)  # fallback: yellow


def _build_virtual_torso_guides(
    pose_draw_points: Dict[int, Tuple[int, int]],
    width: int,
    height: int,
) -> Tuple[Dict[int, Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Add dense torso guide points so shoulder twist is visible even with sparse pose points.
    """
    if 11 not in pose_draw_points or 12 not in pose_draw_points:
        return {}, []

    left_shoulder = np.array(pose_draw_points[11], dtype=np.float32)
    right_shoulder = np.array(pose_draw_points[12], dtype=np.float32)
    shoulder_vec = right_shoulder - left_shoulder
    shoulder_width = float(np.linalg.norm(shoulder_vec))
    if shoulder_width < 2.0:
        return {}, []

    lateral = shoulder_vec / shoulder_width
    neck = (left_shoulder + right_shoulder) * 0.5

    if 23 in pose_draw_points and 24 in pose_draw_points:
        left_hip = np.array(pose_draw_points[23], dtype=np.float32)
        right_hip = np.array(pose_draw_points[24], dtype=np.float32)
    else:
        if 0 in pose_draw_points:
            nose = np.array(pose_draw_points[0], dtype=np.float32)
            down_dir = neck - nose
        else:
            down_dir = np.array([0.0, 1.0], dtype=np.float32)
        dn = float(np.linalg.norm(down_dir))
        if dn < 1e-6:
            down_dir = np.array([0.0, 1.0], dtype=np.float32)
        else:
            down_dir = down_dir / dn
        torso_len = max(shoulder_width * 1.45, 40.0)
        pelvis = neck + down_dir * torso_len
        hip_half = max(shoulder_width * 0.42, 16.0)
        left_hip = pelvis - lateral * hip_half
        right_hip = pelvis + lateral * hip_half

    pelvis = (left_hip + right_hip) * 0.5

    def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a * (1.0 - t) + b * t

    def _valid_int_point(p: np.ndarray) -> Optional[Tuple[int, int]]:
        x = float(p[0])
        y = float(p[1])
        if not _is_valid_pose_point_for_overlay(x, y, width, height):
            return None
        return _clamp_to_int(x, 0, width - 1), _clamp_to_int(y, 0, height - 1)

    points: Dict[int, Tuple[int, int]] = {}
    edges: List[Tuple[int, int]] = []

    base = 1000
    key_points = {
        base + 0: neck,
        base + 1: pelvis,
        base + 2: left_shoulder,
        base + 3: right_shoulder,
        base + 4: left_hip,
        base + 5: right_hip,
    }
    for idx, p in key_points.items():
        ip = _valid_int_point(p)
        if ip is not None:
            points[idx] = ip

    row_ts = [0.2, 0.4, 0.6, 0.8]
    left_rows: List[int] = []
    center_rows: List[int] = []
    right_rows: List[int] = []
    q1_rows: List[int] = []
    q3_rows: List[int] = []

    for i, t in enumerate(row_ts):
        left_p = _lerp(left_shoulder, left_hip, t)
        right_p = _lerp(right_shoulder, right_hip, t)
        center_p = _lerp(neck, pelvis, t)
        q1_p = _lerp(left_p, center_p, 0.5)
        q3_p = _lerp(center_p, right_p, 0.5)

        li = base + 10 + i
        ci = base + 20 + i
        ri = base + 30 + i
        q1i = base + 40 + i
        q3i = base + 50 + i

        for idx, p in [(li, left_p), (ci, center_p), (ri, right_p), (q1i, q1_p), (q3i, q3_p)]:
            ip = _valid_int_point(p)
            if ip is not None:
                points[idx] = ip

        left_rows.append(li)
        center_rows.append(ci)
        right_rows.append(ri)
        q1_rows.append(q1i)
        q3_rows.append(q3i)

    # Outer torso contour and bars.
    contour_left = [base + 2] + left_rows + [base + 4]
    contour_right = [base + 3] + right_rows + [base + 5]
    spine = [base + 0] + center_rows + [base + 1]
    for chain in [contour_left, contour_right, spine]:
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append((a, b))

    # Horizontal rows and diagonal cross-lines to convey twist.
    for li, q1i, ci, q3i, ri in zip(left_rows, q1_rows, center_rows, q3_rows, right_rows):
        edges.extend([(li, q1i), (q1i, ci), (ci, q3i), (q3i, ri), (li, ri)])
    for i in range(len(row_ts) - 1):
        edges.extend([(left_rows[i], right_rows[i + 1]), (right_rows[i], left_rows[i + 1])])

    return points, edges


def _sample_face_points(points: List[Point], limit: int = 5) -> List[Tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points[:limit]]


def _sample_pose_points(points: IndexedPointMap, limit: int = 5) -> Dict[int, Tuple[int, int]]:
    sampled: Dict[int, Tuple[int, int]] = {}
    for idx in sorted(points.keys())[:limit]:
        x, y = points[idx]
        sampled[idx] = (int(round(x)), int(round(y)))
    return sampled


def _estimate_face_center(face_points: List[Point]) -> Point:
    if len(face_points) > 152:
        x = (face_points[6][0] + face_points[152][0]) / 2.0
        y = (face_points[6][1] + face_points[152][1]) / 2.0
        return x, y
    arr = np.array(face_points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _midpoint_of(face_points: List[Point], idx_a: int, idx_b: int) -> np.ndarray:
    if idx_a >= len(face_points) or idx_b >= len(face_points):
        raise ValueError(f"Face landmark index out of range: {idx_a}, {idx_b}")
    return np.array(
        [
            (face_points[idx_a][0] + face_points[idx_b][0]) / 2.0,
            (face_points[idx_a][1] + face_points[idx_b][1]) / 2.0,
        ],
        dtype=np.float32,
    )


def _face_point(face_points: List[Point], idx: int) -> np.ndarray:
    if idx >= len(face_points):
        raise ValueError(f"Face landmark index out of range: {idx}")
    return np.array([face_points[idx][0], face_points[idx][1]], dtype=np.float32)


def get_face_sync_anchors(face_points: List[Point]) -> np.ndarray:
    """
    3-point anchors for affine sync: left-eye center, right-eye center, mouth center.
    """
    left_eye = _midpoint_of(face_points, 33, 133)
    right_eye = _midpoint_of(face_points, 263, 362)
    mouth_center = _midpoint_of(face_points, 13, 14)
    return np.stack([left_eye, right_eye, mouth_center], axis=0).astype(np.float32)


def force_match_pose_face_keypoints(
    pose_points: IndexedPointMap,
    face_points: List[Point],
    enabled: bool = True,
) -> IndexedPointMap:
    """
    Force pose face keypoints to exactly match corresponding face landmarks.
    """
    if not enabled:
        return dict(pose_points)

    out = dict(pose_points)
    mapping = {
        0: _face_point(face_points, 4),       # nose tip
        1: _face_point(face_points, 133),     # left eye inner
        2: _midpoint_of(face_points, 33, 133),  # left eye center
        3: _face_point(face_points, 33),      # left eye outer
        4: _face_point(face_points, 362),     # right eye inner
        5: _midpoint_of(face_points, 263, 362),  # right eye center
        6: _face_point(face_points, 263),     # right eye outer
        7: _face_point(face_points, 234),     # left face edge (ear proxy)
        8: _face_point(face_points, 454),     # right face edge (ear proxy)
        9: _face_point(face_points, 61),      # mouth left
        10: _face_point(face_points, 291),    # mouth right
    }

    for pose_idx, xy in mapping.items():
        if pose_idx not in out:
            continue
        out[pose_idx] = (float(xy[0]), float(xy[1]))
    return out


class PerspectiveFrontalPoseAligner:
    """
    Frontalize face+pose landmarks with 3D rigid transform and perspective projection.
    """

    def __init__(self, width: int, height: int, options: StressTestOptions):
        self.width = width
        self.height = height
        self.options = options
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.max_dim = float(max(width, height))
        self.fx = self.options.camera_focal_scale * self.max_dim
        self.fy = self.options.camera_focal_scale * self.max_dim
        self.base_depth = self.options.camera_base_depth_scale * self.max_dim
        self.K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < eps:
            return np.zeros_like(v)
        return v / n

    def _to_camera_xyz(self, x_px: float, y_px: float, z_rel: float, depth_scale: float) -> np.ndarray:
        X = x_px - self.cx
        Y = y_px - self.cy
        Z = self.base_depth + z_rel * self.max_dim * depth_scale
        Z = float(max(Z, self.options.min_depth_px))
        return np.array([X, Y, Z], dtype=np.float32)

    def _project_camera_xyz(self, p: np.ndarray) -> Point:
        z = float(max(p[2], self.options.min_depth_px))
        u = self.fx * (float(p[0]) / z) + self.cx
        v = self.fy * (float(p[1]) / z) + self.cy
        return float(u), float(v)

    def _solve_anchor_translation(
        self,
        anchor_points_3d: np.ndarray,
        target_anchors_2d: np.ndarray,
    ) -> np.ndarray:
        """
        Solve camera-translation-equivalent shift (tx, ty, tz) so projected anchors fit targets.
        """
        A_rows = []
        b_rows = []
        for p, uv in zip(anchor_points_3d, target_anchors_2d):
            X, Y, Z = float(p[0]), float(p[1]), float(p[2])
            u_t, v_t = float(uv[0]), float(uv[1])
            a = (u_t - self.cx) / self.fx
            b = (v_t - self.cy) / self.fy

            # tx - a*tz = a*Z - X
            A_rows.append([1.0, 0.0, -a])
            b_rows.append(a * Z - X)
            # ty - b*tz = b*Z - Y
            A_rows.append([0.0, 1.0, -b])
            b_rows.append(b * Z - Y)

        A = np.array(A_rows, dtype=np.float32)
        bvec = np.array(b_rows, dtype=np.float32)
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        return sol.astype(np.float32)

    @staticmethod
    def _rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        # y-axis yaw, x-axis pitch, z-axis roll
        Ry = np.array(
            [
                [math.cos(yaw), 0.0, math.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw), 0.0, math.cos(yaw)],
            ],
            dtype=np.float32,
        )
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(pitch), -math.sin(pitch)],
                [0.0, math.sin(pitch), math.cos(pitch)],
            ],
            dtype=np.float32,
        )
        Rz = np.array(
            [
                [math.cos(roll), -math.sin(roll), 0.0],
                [math.sin(roll), math.cos(roll), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return (Rz @ Ry @ Rx).astype(np.float32)

    @staticmethod
    def _matrix_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
        # OpenCV returns (rx, ry, rz) degrees in x/y/z order
        angles, *_ = cv2.RQDecomp3x3(R.astype(np.float64))
        pitch_x, yaw_y, roll_z = angles
        return float(yaw_y), float(pitch_x), float(roll_z)

    def _estimate_torso_rotation(
        self,
        pose_world_points: Optional[IndexedPoint3DMap],
        pose_cam_points: IndexedPoint3DMap,
    ) -> Optional[np.ndarray]:
        def _get_point(src_map: IndexedPoint3DMap, idx: int) -> Optional[np.ndarray]:
            if idx not in src_map:
                return None
            x, y, z = src_map[idx]
            return np.array([x, y, z], dtype=np.float32)

        source = pose_world_points if pose_world_points else pose_cam_points
        p11 = _get_point(source, 11)
        p12 = _get_point(source, 12)
        p23 = _get_point(source, 23)
        p24 = _get_point(source, 24)
        if any(p is None for p in [p11, p12, p23, p24]):
            return None

        neck = (p11 + p12) * 0.5
        pelvis = (p23 + p24) * 0.5
        
        # 11 is person's left shoulder (Right side of image, +X)
        # 12 is person's right shoulder (Left side of image, -X)
        right_axis = self._normalize(p11 - p12)
        # Y points DOWN
        down_axis = self._normalize(pelvis - neck)
        
        forward_axis = self._normalize(np.cross(right_axis, down_axis))
        if np.linalg.norm(forward_axis) < 1e-6:
            return None
        if forward_axis[2] < 0:
            forward_axis = -forward_axis
        down_axis = self._normalize(np.cross(forward_axis, right_axis))
        if np.linalg.norm(down_axis) < 1e-6:
            return None
        return np.stack([right_axis, down_axis, forward_axis], axis=1).astype(np.float32)

    def _estimate_current_rotation(
        self,
        face_points_2d: List[Point],
        face_cam_points: List[np.ndarray],
    ) -> np.ndarray:
        # Canonical 3D head model (generic) for stable PnP pose estimation.
        object_points = np.array(
            [
                [0.0, 0.0, 0.0],       # nose tip
                [0.0, -63.0, -12.0],   # chin
                [-43.0, 32.0, -26.0],  # left eye outer
                [43.0, 32.0, -26.0],   # right eye outer
                [-28.0, -28.0, -20.0], # left mouth corner
                [28.0, -28.0, -20.0],  # right mouth corner
            ],
            dtype=np.float32,
        )
        image_points = np.array(
            [
                face_points_2d[4],
                face_points_2d[152],
                face_points_2d[33],
                face_points_2d[263],
                face_points_2d[61],
                face_points_2d[291],
            ],
            dtype=np.float32,
        )

        ok, rvec, _ = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            np.zeros((4, 1), dtype=np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            return R.astype(np.float32)

        # Fallback: geometry-only estimate from landmark vectors.
        left_eye = (face_cam_points[33] + face_cam_points[133]) * 0.5
        right_eye = (face_cam_points[263] + face_cam_points[362]) * 0.5
        mouth = (face_cam_points[13] + face_cam_points[14]) * 0.5
        eye_center = (left_eye + right_eye) * 0.5
        x_axis = self._normalize(right_eye - left_eye)
        y_seed = self._normalize(mouth - eye_center)
        z_axis = self._normalize(np.cross(x_axis, y_seed))
        if np.linalg.norm(z_axis) < 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if z_axis[2] < 0:
            z_axis = -z_axis
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)

    def frontalize(
        self,
        face_points_3d: List[Point3D],
        pose_points_3d: IndexedPoint3DMap,
        pose_world_points_3d: Optional[IndexedPoint3DMap] = None,
        use_torso_plane_override: Optional[bool] = None,
    ) -> Tuple[List[Point], IndexedPointMap, Dict]:
        face_points_2d = [(x, y) for x, y, _ in face_points_3d]
        face_cam = [
            self._to_camera_xyz(x, y, z, depth_scale=self.options.face_depth_scale)
            for x, y, z in face_points_3d
        ]
        pose_cam = {
            idx: self._to_camera_xyz(x, y, z, depth_scale=self.options.pose_depth_scale)
            for idx, (x, y, z) in pose_points_3d.items()
        }

        R_head_current = self._estimate_current_rotation(face_points_2d, face_cam)
        yaw_before, pitch_before, roll_before = self._matrix_to_ypr(R_head_current)
        R_torso_current = self._estimate_torso_rotation(pose_world_points_3d, pose_cam)

        if self.options.auto_to_frontal:
            yaw_t, pitch_t, roll_t = 0.0, 0.0, 0.0
        else:
            yaw_t = self.options.target_yaw_deg
            pitch_t = self.options.target_pitch_deg
            roll_t = self.options.target_roll_deg
        R_target = self._rotation_from_ypr(yaw_t, pitch_t, roll_t)
        R_head_delta = (R_target @ R_head_current.T).astype(np.float32)
        use_torso_plane = self.options.use_torso_plane_facing
        if use_torso_plane_override is not None:
            use_torso_plane = use_torso_plane_override
        if use_torso_plane and R_torso_current is not None:
            R_pose_delta = (R_target @ R_torso_current.T).astype(np.float32)
            if self.options.torso_facing_strength < 0.999:
                rvec, _ = cv2.Rodrigues(R_pose_delta.astype(np.float64))
                rvec = rvec * float(max(0.0, min(1.0, self.options.torso_facing_strength)))
                R_pose_delta, _ = cv2.Rodrigues(rvec)
                R_pose_delta = R_pose_delta.astype(np.float32)
        else:
            R_pose_delta = R_head_delta

        pivot = face_cam[4].copy()  # nose tip
        face_rot = [R_head_delta @ (p - pivot) + pivot for p in face_cam]
        pose_pivot = pivot
        pose_rot = {
            idx: R_pose_delta @ (p - pose_pivot) + pose_pivot for idx, p in pose_cam.items()
        }

        target_anchors = _make_frontal_target_anchors(self.width, self.height, self.options)
        left_eye_rot = (face_rot[33] + face_rot[133]) * 0.5
        right_eye_rot = (face_rot[263] + face_rot[362]) * 0.5
        mouth_rot = (face_rot[13] + face_rot[14]) * 0.5
        anchor_3d = np.stack([left_eye_rot, right_eye_rot, mouth_rot], axis=0).astype(np.float32)
        tx, ty, tz = self._solve_anchor_translation(anchor_3d, target_anchors)
        trans = np.array([tx, ty, tz], dtype=np.float32)
        face_rot = [p + trans for p in face_rot]
        pose_rot = {idx: p + trans for idx, p in pose_rot.items()}

        face_2d = [self._project_camera_xyz(p) for p in face_rot]
        pose_2d = {idx: self._project_camera_xyz(p) for idx, p in pose_rot.items()}

        projected_anchors = get_face_sync_anchors(face_2d)
        reprojection_error = float(
            np.mean(np.linalg.norm(projected_anchors - target_anchors, axis=1))
        )

        R_head_after = (R_head_delta @ R_head_current).astype(np.float32)
        yaw_after, pitch_after, roll_after = self._matrix_to_ypr(R_head_after)
        if R_torso_current is not None:
            R_torso_after = (R_pose_delta @ R_torso_current).astype(np.float32)
            torso_yaw_after, torso_pitch_after, torso_roll_after = self._matrix_to_ypr(R_torso_after)
            torso_yaw_before, torso_pitch_before, torso_roll_before = self._matrix_to_ypr(R_torso_current)
        else:
            torso_yaw_before = torso_pitch_before = torso_roll_before = None
            torso_yaw_after = torso_pitch_after = torso_roll_after = None
        debug = {
            "ypr_before": {
                "yaw": round(yaw_before, 3),
                "pitch": round(pitch_before, 3),
                "roll": round(roll_before, 3),
            },
            "ypr_target": {
                "yaw": round(yaw_t, 3),
                "pitch": round(pitch_t, 3),
                "roll": round(roll_t, 3),
            },
            "ypr_after": {
                "yaw": round(yaw_after, 3),
                "pitch": round(pitch_after, 3),
                "roll": round(roll_after, 3),
            },
            "torso_ypr_before": None if torso_yaw_before is None else {
                "yaw": round(torso_yaw_before, 3),
                "pitch": round(torso_pitch_before, 3),
                "roll": round(torso_roll_before, 3),
            },
            "torso_ypr_after": None if torso_yaw_after is None else {
                "yaw": round(torso_yaw_after, 3),
                "pitch": round(torso_pitch_after, 3),
                "roll": round(torso_roll_after, 3),
            },
            "use_torso_plane_facing": bool(use_torso_plane and R_torso_current is not None),
            "camera": {
                "fx": round(self.fx, 3),
                "fy": round(self.fy, 3),
                "cx": round(self.cx, 3),
                "cy": round(self.cy, 3),
                "base_depth": round(self.base_depth, 3),
            },
            "reprojection_error_px": round(reprojection_error, 3),
        }
        return face_2d, pose_2d, debug

def _make_frontal_target_anchors(
    width: int,
    height: int,
    stress_options: StressTestOptions,
) -> np.ndarray:
    yaw_deg = 0.0 if stress_options.auto_to_frontal else float(stress_options.target_yaw_deg)
    yaw_deg = float(np.clip(yaw_deg, -90.0, 90.0))
    yaw_rad = math.radians(yaw_deg)

    cx = float(width * stress_options.frontal_center_x_ratio)
    cx += float(width * 0.12 * math.sin(yaw_rad))
    eye_y = float(height * stress_options.frontal_eye_y_ratio)
    mouth_y = float(height * stress_options.frontal_mouth_y_ratio)
    # Keep eye distance finite even at +-90 to avoid degenerate affine.
    eye_scale = max(0.35, abs(math.cos(yaw_rad)))
    eye_half_dist = float(width * stress_options.frontal_eye_half_dist_ratio * eye_scale)
    mouth_x = float(cx + width * 0.05 * math.sin(yaw_rad))
    return np.array(
        [
            [cx - eye_half_dist, eye_y],  # left eye center
            [cx + eye_half_dist, eye_y],  # right eye center
            [mouth_x, mouth_y],  # mouth center with yaw shift
        ],
        dtype=np.float32,
    )


def _make_yaw_target_anchors_from_src(src: np.ndarray, yaw_deg: float) -> np.ndarray:
    """
    Build yaw targets from current anchors to preserve face scale/position.
    yaw=0 -> identity mapping (no enlargement).
    """
    # Interpret yaw_deg as camera yaw movement around a fixed person.
    # Camera moves left(-): person appears to look right(+), so invert sign.
    yaw = -float(np.clip(yaw_deg, -90.0, 90.0))
    if abs(yaw) < 1e-6:
        return src.astype(np.float32).copy()

    left_eye = src[0].astype(np.float32)
    right_eye = src[1].astype(np.float32)
    mouth = src[2].astype(np.float32)
    eye_center = (left_eye + right_eye) * 0.5
    eye_half = max(float(np.linalg.norm(right_eye - left_eye)) * 0.5, 8.0)
    yaw_rad = math.radians(yaw)
    cos_abs = abs(math.cos(yaw_rad))
    sin_v = math.sin(yaw_rad)

    # Shrink horizontal eye spread as yaw increases, but keep finite at +-90.
    eye_half_t = eye_half * max(0.30, cos_abs)
    # Use stronger camera-motion cue so left/right direction is visually unambiguous.
    center_x_shift = eye_half * 1.55 * sin_v
    mouth_x_shift = eye_half * 1.25 * sin_v

    eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
    cx = float(eye_center[0] + center_x_shift)
    mouth_x = float(eye_center[0] + mouth_x_shift)

    return np.array(
        [
            [cx - eye_half_t, eye_y],
            [cx + eye_half_t, eye_y],
            [mouth_x, float(mouth[1])],
        ],
        dtype=np.float32,
    )


def _apply_affine_to_face_points(face_points: List[Point], affine_mat: np.ndarray) -> List[Point]:
    out: List[Point] = []
    for x, y in face_points:
        x2 = affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2]
        y2 = affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2]
        out.append((float(x2), float(y2)))
    return out


def _apply_affine_to_pose_points(pose_points: IndexedPointMap, affine_mat: np.ndarray) -> IndexedPointMap:
    out: IndexedPointMap = {}
    for idx, (x, y) in pose_points.items():
        x2 = affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2]
        y2 = affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2]
        out[idx] = (float(x2), float(y2))
    return out


def _clip_face_points(face_points: List[Point], width: int, height: int) -> List[Point]:
    return [
        (
            float(np.clip(x, 0.0, float(width - 1))),
            float(np.clip(y, 0.0, float(height - 1))),
        )
        for x, y in face_points
    ]


def _clip_pose_points(pose_points: IndexedPointMap, width: int, height: int) -> IndexedPointMap:
    out: IndexedPointMap = {}
    for idx, (x, y) in pose_points.items():
        out[idx] = (
            float(np.clip(x, 0.0, float(width - 1))),
            float(np.clip(y, 0.0, float(height - 1))),
        )
    return out


def apply_frontal_affine_to_face_and_pose(
    face_points: List[Point],
    pose_points: IndexedPointMap,
    width: int,
    height: int,
    stress_options: StressTestOptions,
) -> Tuple[List[Point], IndexedPointMap, bool]:
    if not stress_options.apply_frontal_affine:
        return face_points, pose_points, False

    src = get_face_sync_anchors(face_points).astype(np.float32)
    if stress_options.auto_to_frontal:
        dst = _make_frontal_target_anchors(width, height, stress_options)
    else:
        dst = _make_yaw_target_anchors_from_src(src, stress_options.target_yaw_deg)

    # Guard against degenerate anchors
    src_area = abs(np.cross(src[1] - src[0], src[2] - src[0]))
    if src_area < 1e-6:
        return face_points, pose_points, False

    affine_mat = cv2.getAffineTransform(src, dst)
    face_out = _clip_face_points(_apply_affine_to_face_points(face_points, affine_mat), width, height)
    pose_out = _clip_pose_points(_apply_affine_to_pose_points(pose_points, affine_mat), width, height)
    return face_out, pose_out, True


def get_face_bbox_from_landmarks(
    face_points: List[Point],
    width: int,
    height: int,
    padding_ratio: float = 0.15,
) -> Tuple[int, int, int, int]:
    """
    얼굴 랜드마크의 외접(bounding box)을 계산하고 padding을 적용해 (x, y, w, h) 반환.
    """
    if not face_points:
        raise ValueError("Face points are empty.")
    arr = np.array(face_points, dtype=np.float32)
    min_x, min_y = float(arr[:, 0].min()), float(arr[:, 1].min())
    max_x, max_y = float(arr[:, 0].max()), float(arr[:, 1].max())
    pad_w = (max_x - min_x) * padding_ratio
    pad_h = (max_y - min_y) * padding_ratio
    x1 = max(0, int(round(min_x - pad_w)))
    y1 = max(0, int(round(min_y - pad_h)))
    x2 = min(width, int(round(max_x + pad_w)))
    y2 = min(height, int(round(max_y + pad_h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face bounding box.")
    return x1, y1, x2 - x1, y2 - y1


def save_face_org_from_landmarks(
    image_bgr: np.ndarray,
    face_points: List[Point],
    save_path: str,
    padding_ratio: float = 0.15,
) -> str:
    """
    얼굴 랜드마크 외접 영역을 잘라 임시 이미지(face_org)로 저장한다.
    Returns: 저장된 파일 경로.
    """
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = get_face_bbox_from_landmarks(face_points, w, h, padding_ratio=padding_ratio)
    crop = image_bgr[y : y + bh, x : x + bw]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, crop)
    return save_path


def extract_face_landmarks(detector: T1LandmarkDetector, image_bgr: np.ndarray) -> List[Point]:
    points, _ = detector.get_landmarks(image_bgr, apply_filter=False)
    if not points:
        raise ValueError("Face landmarks not detected.")
    return [(float(x), float(y)) for x, y in points]


def extract_pose_landmarks(detector: T1LandmarkDetector, image_bgr: np.ndarray) -> IndexedPointMap:
    pose_results = detector.pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not pose_results.pose_landmarks:
        raise ValueError("Pose landmarks not detected.")

    h, w = image_bgr.shape[:2]
    points: IndexedPointMap = {}
    for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
        points[idx] = (lm.x * w, lm.y * h)
    return points


def extract_hand_landmarks(
    image_bgr: np.ndarray,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
) -> Tuple[List[Point], List[Point]]:
    """
    MediaPipe Hands로 양손 21개 랜드마크 추출. (left_hand_21pts, right_hand_21pts), 없으면 [].
    """
    if _mp_hands is None:
        return [], []
    h, w = image_bgr.shape[:2]
    hands = _mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    )
    try:
        results = hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        left_pts: List[Point] = []
        right_pts: List[Point] = []
        if not results.multi_hand_landmarks:
            return left_pts, right_pts
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness or [],
        ):
            pts = [(float(lm.x * w), float(lm.y * h)) for lm in hand_landmarks.landmark]
            if handedness.classification[0].label == "Left":
                left_pts = pts
            else:
                right_pts = pts
        return left_pts, right_pts
    finally:
        hands.close()


def extract_face_landmarks_3d(detector: T1LandmarkDetector, image_bgr: np.ndarray) -> List[Point3D]:
    results = detector.face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("Face landmarks not detected.")
    h, w = image_bgr.shape[:2]
    points: List[Point3D] = []
    for lm in results.multi_face_landmarks[0].landmark:
        points.append((float(lm.x * w), float(lm.y * h), float(lm.z)))
    return points


def extract_pose_landmarks_3d(detector: T1LandmarkDetector, image_bgr: np.ndarray) -> IndexedPoint3DMap:
    pose_results = detector.pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not pose_results.pose_landmarks:
        raise ValueError("Pose landmarks not detected.")
    h, w = image_bgr.shape[:2]
    points: IndexedPoint3DMap = {}
    for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
        points[idx] = (float(lm.x * w), float(lm.y * h), float(lm.z))
    return points


def extract_pose_world_landmarks_3d(detector: T1LandmarkDetector, image_bgr: np.ndarray) -> Optional[IndexedPoint3DMap]:
    pose_results = detector.pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not pose_results.pose_world_landmarks:
        return None
    points: IndexedPoint3DMap = {}
    for idx, lm in enumerate(pose_results.pose_world_landmarks.landmark):
        points[idx] = (float(lm.x), float(lm.y), float(lm.z))
    return points


def symmetrize_face_landmarks(face_points: List[Point]) -> Tuple[List[Point], float]:
    midline_x = float(compute_robust_midline(face_points))
    sym_points = symmetrize_landmarks(face_points, midline_x)
    return [(float(x), float(y)) for x, y in sym_points], midline_x


def _fallback_pose_midline(pose_points: IndexedPointMap) -> float:
    if 11 in pose_points and 12 in pose_points:
        return (pose_points[11][0] + pose_points[12][0]) / 2.0
    if 23 in pose_points and 24 in pose_points:
        return (pose_points[23][0] + pose_points[24][0]) / 2.0
    if pose_points:
        xs = [xy[0] for xy in pose_points.values()]
        return float(np.mean(xs))
    raise ValueError("Cannot estimate pose midline from empty pose landmarks.")


def symmetrize_pose_landmarks(pose_points: IndexedPointMap, midline_x: Optional[float] = None) -> IndexedPointMap:
    if midline_x is None:
        midline_x = _fallback_pose_midline(pose_points)

    sym_points: IndexedPointMap = dict(pose_points)
    for left_idx, right_idx in POSE_LEFT_RIGHT_PAIRS:
        if left_idx not in sym_points or right_idx not in sym_points:
            continue

        l_x, l_y = sym_points[left_idx]
        r_x, r_y = sym_points[right_idx]

        l_dist = midline_x - l_x
        r_dist = r_x - midline_x
        avg_dist = (l_dist + r_dist) / 2.0
        avg_y = (l_y + r_y) / 2.0

        sym_points[left_idx] = (midline_x - avg_dist, avg_y)
        sym_points[right_idx] = (midline_x + avg_dist, avg_y)

    for center_idx in POSE_CENTER_INDICES:
        if center_idx in sym_points:
            _, y = sym_points[center_idx]
            sym_points[center_idx] = (midline_x, y)

    return sym_points


def apply_stress_to_face_landmarks(
    face_points: List[Point],
    image_width: int,
    image_height: int,
    stress_options: StressTestOptions,
) -> List[Point]:
    if not stress_options.enabled:
        return face_points

    pts = np.array(face_points, dtype=np.float32)
    cx, cy = _estimate_face_center(face_points)

    reverse_strength = float(np.clip(stress_options.face_reverse_strength, 0.0, 1.2))
    if reverse_strength > 0.0:
        mirrored_x = 2.0 * cx - pts[:, 0]
        pts[:, 0] = pts[:, 0] * (1.0 - reverse_strength) + mirrored_x * reverse_strength

    pts[:, 0] += stress_options.face_x_shift_ratio * image_width
    pts[:, 1] += stress_options.face_y_shift_ratio * image_height

    if abs(stress_options.face_shear) > 1e-8:
        pts[:, 0] += stress_options.face_shear * (pts[:, 1] - cy)

    if abs(stress_options.face_roll_deg) > 1e-8:
        theta = math.radians(stress_options.face_roll_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x = pts[:, 0] - cx
        y = pts[:, 1] - cy
        pts[:, 0] = x * cos_t - y * sin_t + cx
        pts[:, 1] = x * sin_t + y * cos_t + cy

    pts[:, 0] = np.clip(pts[:, 0], 0, image_width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, image_height - 1)
    return [(float(x), float(y)) for x, y in pts]


def _set_arm_direction(
    pose_points: IndexedPointMap,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    hand_indices: List[int],
    angle_deg: float,
    arm_extension: float,
    hand_spread_px: float,
) -> None:
    required = [shoulder_idx, elbow_idx, wrist_idx]
    if any(idx not in pose_points for idx in required):
        return

    shoulder = np.array(pose_points[shoulder_idx], dtype=np.float32)
    elbow = np.array(pose_points[elbow_idx], dtype=np.float32)
    wrist = np.array(pose_points[wrist_idx], dtype=np.float32)

    upper_len = max(float(np.linalg.norm(elbow - shoulder)), 10.0) * arm_extension
    lower_len = max(float(np.linalg.norm(wrist - elbow)), 10.0) * arm_extension

    theta = math.radians(angle_deg)
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    new_elbow = shoulder + direction * upper_len
    new_wrist = new_elbow + direction * lower_len

    pose_points[elbow_idx] = (float(new_elbow[0]), float(new_elbow[1]))
    pose_points[wrist_idx] = (float(new_wrist[0]), float(new_wrist[1]))

    offsets = [-1.0, 0.0, 1.0]
    for offset, hand_idx in zip(offsets, hand_indices):
        if hand_idx not in pose_points:
            continue
        target = new_wrist + perp * (offset * hand_spread_px)
        pose_points[hand_idx] = (float(target[0]), float(target[1]))


def apply_stress_to_pose_landmarks(
    pose_points: IndexedPointMap,
    image_width: int,
    image_height: int,
    stress_options: StressTestOptions,
    face_sync_src_anchors: Optional[np.ndarray] = None,
    face_sync_dst_anchors: Optional[np.ndarray] = None,
    allowed_source_indices: Optional[Set[int]] = None,
) -> IndexedPointMap:
    if not stress_options.enabled:
        return dict(pose_points)

    out_points: IndexedPointMap = dict(pose_points)

    if (
        stress_options.sync_pose_face_with_eye_mouth
        and face_sync_src_anchors is not None
        and face_sync_dst_anchors is not None
    ):
        affine_mat = cv2.getAffineTransform(
            face_sync_src_anchors.astype(np.float32),
            face_sync_dst_anchors.astype(np.float32),
        )
        for idx in POSE_FACE_SYNC_INDICES:
            if idx not in out_points:
                continue
            x, y = out_points[idx]
            x2 = affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2]
            y2 = affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2]
            out_points[idx] = (float(x2), float(y2))

    if stress_options.left_arm_angle_deg is not None:
        left_required = {11, 13, 15, 17, 19, 21}
        if allowed_source_indices is not None and not left_required.issubset(allowed_source_indices):
            pass
        else:
            _set_arm_direction(
                pose_points=out_points,
                shoulder_idx=11,
                elbow_idx=13,
                wrist_idx=15,
                hand_indices=[17, 19, 21],
                angle_deg=stress_options.left_arm_angle_deg,
                arm_extension=stress_options.arm_extension,
                hand_spread_px=stress_options.hand_spread_px,
            )

    if stress_options.right_arm_angle_deg is not None:
        right_required = {12, 14, 16, 18, 20, 22}
        if allowed_source_indices is not None and not right_required.issubset(allowed_source_indices):
            pass
        else:
            _set_arm_direction(
                pose_points=out_points,
                shoulder_idx=12,
                elbow_idx=14,
                wrist_idx=16,
                hand_indices=[18, 20, 22],
                angle_deg=stress_options.right_arm_angle_deg,
                arm_extension=stress_options.arm_extension,
                hand_spread_px=stress_options.hand_spread_px,
            )

    for idx, (x, y) in out_points.items():
        out_points[idx] = (float(x), float(y))
    return out_points


def _draw_hand_overlay(
    overlay: np.ndarray,
    hand_points: List[Point],
    width: int,
    height: int,
    hand_radius: int = 4,
    line_thickness: int = 1,
) -> None:
    """손 21점 + HAND_CONNECTIONS 그리기."""
    if not hand_points or not HAND_CONNECTIONS:
        return
    pts_int: Dict[int, Tuple[int, int]] = {}
    for idx, (x, y) in enumerate(hand_points):
        if not _is_valid_pose_point_for_overlay(x, y, width, height):
            continue
        px = _clamp_to_int(x, 0, width - 1)
        py = _clamp_to_int(y, 0, height - 1)
        pts_int[idx] = (px, py)
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx not in pts_int or end_idx not in pts_int:
            continue
        cv2.line(
            overlay,
            pts_int[start_idx],
            pts_int[end_idx],
            (255, 255, 255),
            line_thickness,
            lineType=cv2.LINE_8,
        )
    for (px, py) in pts_int.values():
        cv2.circle(overlay, (px, py), hand_radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_8)
        cv2.circle(overlay, (px, py), hand_radius, (255, 200, 100), -1, lineType=cv2.LINE_8)


def render_overlay(
    width: int,
    height: int,
    face_points: List[Point],
    pose_points: IndexedPointMap,
    face_draw_indices: Optional[Set[int]] = None,
    pose_allowed_indices: Optional[Set[int]] = None,
    hand_points_left: Optional[List[Point]] = None,
    hand_points_right: Optional[List[Point]] = None,
    face_radius: int = 3,
    pose_radius: int = 7,
) -> np.ndarray:
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, (x, y) in enumerate(face_points):
        if face_draw_indices is not None and idx not in face_draw_indices:
            continue
        px = _clamp_to_int(x, 0, width - 1)
        py = _clamp_to_int(y, 0, height - 1)
        cv2.circle(overlay, (px, py), face_radius, (255, 255, 255), -1, lineType=cv2.LINE_8)

    pose_draw_points: Dict[int, Tuple[int, int]] = {}
    for idx, (x, y) in pose_points.items():
        if pose_allowed_indices is not None and idx not in pose_allowed_indices:
            continue
        if not _is_valid_pose_point_for_overlay(x, y, width, height):
            continue
        px = _clamp_to_int(x, 0, width - 1)
        py = _clamp_to_int(y, 0, height - 1)
        pose_draw_points[idx] = (px, py)

    for start_idx, end_idx in POSE_DRAW_CONNECTIONS:
        if start_idx not in pose_draw_points or end_idx not in pose_draw_points:
            continue
        cv2.line(
            overlay,
            pose_draw_points[start_idx],
            pose_draw_points[end_idx],
            (255, 255, 255),
            2,
            lineType=cv2.LINE_8,
        )

    virtual_torso_points, virtual_torso_edges = _build_virtual_torso_guides(
        pose_draw_points=pose_draw_points,
        width=width,
        height=height,
    )
    for start_idx, end_idx in virtual_torso_edges:
        if start_idx not in virtual_torso_points or end_idx not in virtual_torso_points:
            continue
        cv2.line(
            overlay,
            virtual_torso_points[start_idx],
            virtual_torso_points[end_idx],
            (255, 255, 255),
            2,
            lineType=cv2.LINE_8,
        )

    for idx, (px, py) in pose_draw_points.items():
        pose_color = _get_pose_color(idx)
        cv2.circle(overlay, (px, py), pose_radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_8)
        cv2.circle(overlay, (px, py), pose_radius, pose_color, -1, lineType=cv2.LINE_8)

    virtual_radius = max(3, pose_radius - 2)
    for idx, (px, py) in virtual_torso_points.items():
        pose_color = _get_pose_color(idx)
        cv2.circle(overlay, (px, py), virtual_radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_8)
        cv2.circle(overlay, (px, py), virtual_radius, pose_color, -1, lineType=cv2.LINE_8)

    # 손가락 랜드마크(21점) + 연결선 오버레이
    if hand_points_left:
        _draw_hand_overlay(overlay, hand_points_left, width, height)
    if hand_points_right:
        _draw_hand_overlay(overlay, hand_points_right, width, height)

    return overlay


def build_idphoto_landmark_overlay(
    input_image_path: str,
    overlay_output_path: str,
    stress_options: Optional[StressTestOptions] = None,
    use_raw_detected_points: bool = False,
) -> Dict:
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {input_image_path}")

    stress_options = stress_options or StressTestOptions()
    frontal_debug: Optional[Dict] = None
    frontal_3d_applied = False
    frontal_affine_applied = False

    detector = T1LandmarkDetector()
    try:
        face_points_before = extract_face_landmarks(detector, image_bgr)
        pose_points_before = extract_pose_landmarks(detector, image_bgr)
        face_points_3d_before = extract_face_landmarks_3d(detector, image_bgr)
        pose_points_3d_before = extract_pose_landmarks_3d(detector, image_bgr)
        pose_world_points_3d_before = extract_pose_world_landmarks_3d(detector, image_bgr)

        h, w = image_bgr.shape[:2]
        # 얼굴 랜드마크 외접을 임시 이미지(face_org)로 저장
        face_org_path = str(
            Path(overlay_output_path).parent / f"{Path(input_image_path).stem}_face_org.png"
        )
        save_face_org_from_landmarks(image_bgr, face_points_before, face_org_path)

        if use_raw_detected_points:
            midline_x = float("nan")
            face_points_after = list(face_points_before)
            pose_points_after = dict(pose_points_before)
            pose_allowed_indices = {
                idx
                for idx, (x, y) in pose_points_before.items()
                if _is_valid_pose_point_for_overlay(x, y, w, h)
            }
        else:
            try:
                # 3D Path: Handles both auto_to_frontal and specific camera yaw/pitch/roll natively in 3D
                aligner = PerspectiveFrontalPoseAligner(width=w, height=h, options=stress_options)
                face_points_after, pose_points_after, frontal_debug = aligner.frontalize(
                    face_points_3d=face_points_3d_before,
                    pose_points_3d=pose_points_3d_before,
                    pose_world_points_3d=pose_world_points_3d_before,
                )
                if (
                    frontal_debug is not None
                    and stress_options.auto_to_frontal
                    and frontal_debug.get("reprojection_error_px") is not None
                    and frontal_debug["reprojection_error_px"] > stress_options.max_reprojection_error_px
                ):
                    raise ValueError(
                        f"High reprojection error: {frontal_debug['reprojection_error_px']}px"
                    )
                frontal_3d_applied = True

                midline_x = float(compute_robust_midline(face_points_after))
                pose_allowed_indices = {
                    idx
                    for idx, (x, y) in pose_points_after.items()
                    if _is_valid_pose_point_for_overlay(x, y, w, h)
                }
                if len(pose_allowed_indices) < 3 and frontal_debug is not None:
                    frontal_debug["low_pose_visibility_before_sync"] = len(pose_allowed_indices)
            except Exception as exc:
                # Fallback to legacy 2D path when 3D frontalization is unstable.
                face_points_symmetric, midline_x = symmetrize_face_landmarks(face_points_before)
                pose_points_symmetric = symmetrize_pose_landmarks(pose_points_before, midline_x=midline_x)
                pose_allowed_indices = {
                    idx
                    for idx, (x, y) in pose_points_symmetric.items()
                    if _is_valid_pose_point_for_overlay(x, y, w, h)
                }
                face_points_after = apply_stress_to_face_landmarks(
                    face_points_symmetric,
                    image_width=w,
                    image_height=h,
                    stress_options=stress_options,
                )
                face_sync_src_anchors = get_face_sync_anchors(face_points_symmetric)
                face_sync_dst_anchors = get_face_sync_anchors(face_points_after)
                pose_points_after = apply_stress_to_pose_landmarks(
                    pose_points_symmetric,
                    image_width=w,
                    image_height=h,
                    stress_options=stress_options,
                    face_sync_src_anchors=face_sync_src_anchors,
                    face_sync_dst_anchors=face_sync_dst_anchors,
                    allowed_source_indices=pose_allowed_indices,
                )
                if not stress_options.auto_to_frontal:
                    # Step 1) lock fallback output to canonical frontal(0) baseline.
                    neutral_affine_options = StressTestOptions(**asdict(stress_options))
                    neutral_affine_options.auto_to_frontal = True
                    neutral_affine_options.target_yaw_deg = 0.0
                    neutral_affine_options.target_pitch_deg = 0.0
                    neutral_affine_options.target_roll_deg = 0.0
                    face_points_after, pose_points_after, _ = apply_frontal_affine_to_face_and_pose(
                        face_points=face_points_after,
                        pose_points=pose_points_after,
                        width=w,
                        height=h,
                        stress_options=neutral_affine_options,
                    )
                    # Step 2) apply camera-yaw from frontal baseline.
                    face_points_after, pose_points_after, frontal_affine_applied = apply_frontal_affine_to_face_and_pose(
                        face_points=face_points_after,
                        pose_points=pose_points_after,
                        width=w,
                        height=h,
                        stress_options=stress_options,
                    )
                frontal_debug = {
                    "fallback_reason": str(exc),
                    "reprojection_error_px": None,
                }

        if not use_raw_detected_points:
            # For frontal target (auto or yaw=0), enforce bilateral symmetry on central nose line.
            if stress_options.auto_to_frontal or abs(float(stress_options.target_yaw_deg)) < 1e-6:
                face_points_after, midline_x = symmetrize_face_landmarks(face_points_after)
                for center_idx in [4, 6, 168, 1, 2, 5, 195, 197]:
                    if center_idx < len(face_points_after):
                        _, cy = face_points_after[center_idx]
                        face_points_after[center_idx] = (float(midline_x), float(cy))

        pose_points_after = force_match_pose_face_keypoints(
            pose_points_after,
            face_points_after,
            # Even in raw_detected mode, align pose facial keypoints to face landmarks
            # so eyes/mouth are spatially consistent for overlay conditioning.
            enabled=stress_options.force_pose_face_keypoint_match,
        )
        pose_allowed_indices = {
            idx
            for idx, (x, y) in pose_points_after.items()
            if _is_valid_pose_point_for_overlay(x, y, w, h)
        }

        hand_points_left, hand_points_right = extract_hand_landmarks(image_bgr)
        overlay_rgb = render_overlay(
            width=w,
            height=h,
            face_points=face_points_after,
            pose_points=pose_points_after,
            face_draw_indices=MAJOR_FACE_LANDMARK_INDICES,
            pose_allowed_indices=pose_allowed_indices,
            hand_points_left=hand_points_left,
            hand_points_right=hand_points_right,
        )

        output_path = Path(overlay_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_rgb, mode="RGB").save(output_path)

        return {
            "overlay_output_path": str(output_path),
            "face_org_path": face_org_path,
            "face_count": len(face_points_after),
            "face_draw_count": len([idx for idx in MAJOR_FACE_LANDMARK_INDICES if idx < len(face_points_after)]),
            "pose_count": len(pose_points_after),
            "pose_allowed_count": len(pose_allowed_indices),
            "midline_x": None if use_raw_detected_points else round(midline_x, 2),
            "use_raw_detected_points": use_raw_detected_points,
            "stress_enabled": stress_options.enabled,
            "stress_options": asdict(stress_options),
            "frontal_affine_applied": frontal_affine_applied,
            "frontal_3d_applied": frontal_3d_applied,
            "frontal_debug": frontal_debug,
            "pose_world_available": pose_world_points_3d_before is not None,
            "pose_face_sync_applied": bool(
                stress_options.enabled and stress_options.sync_pose_face_with_eye_mouth
            ),
            "face_before_sample": _sample_face_points(face_points_before),
            "face_after_sample": _sample_face_points(face_points_after),
            "pose_before_sample": _sample_pose_points(pose_points_before),
            "pose_after_sample": _sample_pose_points(pose_points_after),
        }
    finally:
        del detector
