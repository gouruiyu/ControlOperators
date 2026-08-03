from pathlib import Path
import numpy as np
import os, urllib.request, zipfile, argparse, time, json
import torch
import torch.nn as nn
from tqdm import tqdm
import quat
import bvh
from typing import List, Tuple

from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage

import control_operators as co
import control_encoder
import networks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# Database Utils
# ----------------------------

# Bone weights taken from:
# https://theorangeduck.com/page/joint-error-propagation
weights_mesh = {
    "Simulation":       0.00000000,
    "Hips":             0.27088639,
    "Spine":            0.12776886,
    "Spine1":           0.10730254,
    "Spine2":           0.08733685,
    "Spine3":           0.07508411,
    "Neck":             0.00838600,
    "Neck1":            0.00639638,
    "Head":             0.00515253,
    "HeadEnd":          0.00063045,
    "RightShoulder":    0.02654437,
    "RightArm":         0.02060832,
    "RightForeArm":     0.00825604,
    "RightHand":        0.00213240,
    "RightHandThumb1":  0.00073802,
    "RightHandThumb2":  0.00066565,
    "RightHandThumb3":  0.00063558,
    "RightHandThumb4":  0.00063045,
    "RightHandIndex1":  0.00070377,
    "RightHandIndex2":  0.00064898,
    "RightHandIndex3":  0.00063289,
    "RightHandIndex4":  0.00063045,
    "RightHandMiddle1": 0.00072178,
    "RightHandMiddle2": 0.00065547,
    "RightHandMiddle3": 0.00063321,
    "RightHandMiddle4": 0.00063045,
    "RightHandRing1":   0.00070793,
    "RightHandRing2":   0.00065231,
    "RightHandRing3":   0.00063322,
    "RightHandRing4":   0.00063045,
    "RightHandPinky1":  0.00067184,
    "RightHandPinky2":  0.00063829,
    "RightHandPinky3":  0.00063110,
    "RightHandPinky4":  0.00063045,
    "RightForeArmEnd":  0.00063045,
    "RightArmEnd":      0.00063045,
    "LeftShoulder":     0.02739252,
    "LeftArm":          0.02113067,
    "LeftForeArm":      0.00849728,
    "LeftHand":         0.00210641,
    "LeftHandThumb1":   0.00071845,
    "LeftHandThumb2":   0.00065790,
    "LeftHandThumb3":   0.00063489,
    "LeftHandThumb4":   0.00063045,
    "LeftHandIndex1":   0.00069211,
    "LeftHandIndex2":   0.00064446,
    "LeftHandIndex3":   0.00063293,
    "LeftHandIndex4":   0.00063045,
    "LeftHandMiddle1":  0.00071069,
    "LeftHandMiddle2":  0.00065042,
    "LeftHandMiddle3":  0.00063314,
    "LeftHandMiddle4":  0.00063045,
    "LeftHandRing1":    0.00070524,
    "LeftHandRing2":    0.00065236,
    "LeftHandRing3":    0.00063302,
    "LeftHandRing4":    0.00063045,
    "LeftHandPinky1":   0.00067250,
    "LeftHandPinky2":   0.00064092,
    "LeftHandPinky3":   0.00063160,
    "LeftHandPinky4":   0.00063045,
    "LeftForeArmEnd":   0.00063045,
    "LeftArmEnd":       0.00063045,
    "RightUpLeg":       0.05690333,
    "RightLeg":         0.02043630,
    "RightFoot":        0.00305942,
    "RightToeBase":     0.00080056,
    "RightToeBaseEnd":  0.00063045,
    "RightLegEnd":      0.00063045,
    "RightUpLegEnd":    0.00063045,
    "LeftUpLeg":        0.05668447,
    "LeftLeg":          0.02033588,
    "LeftFoot":         0.00289429,
    "LeftToeBase":      0.00078392,
    "LeftToeBaseEnd":   0.00063045,
    "LeftLegEnd":       0.00063045,
    "LeftUpLegEnd":     0.00063045
}

tags_data = [                                                                  
    ('aiming1_subject1', 'all', 0, 14367),
    ('aiming1_subject4', 'all', 0, 14367),
    ('aiming2_subject2', 'all', 0, 18343),
    ('aiming2_subject3', 'all', 0, 18343),
    ('aiming2_subject5', 'all', 0, 18343),
    ('dance1_subject1', 'all', 0, 7889),
    ('dance1_subject2', 'all', 0, 7889),
    ('dance1_subject3', 'all', 0, 7889),
    ('dance2_subject1', 'all', 0, 13541),
    ('dance2_subject2', 'all', 0, 13541),
    ('dance2_subject3', 'all', 0, 13541),
    ('dance2_subject4', 'all', 0, 13541),
    ('dance2_subject5', 'all', 0, 13541),
    ('fight1_subject2', 'all', 0, 14693),
    ('fight1_subject3', 'all', 0, 14693),
    ('fight1_subject5', 'all', 0, 14693),
    ('fightAndSports1_subject1', 'all', 0, 14709),
    ('fightAndSports1_subject4', 'all', 0, 14709),
    ('jumps1_subject1', 'all', 0, 9835),
    ('jumps1_subject2', 'all', 0, 9835),
    ('jumps1_subject5', 'all', 0, 9835),
    ('run1_subject2', 'all', 0, 14269),
    ('run1_subject5', 'all', 0, 14269),
    ('run2_subject1', 'all', 0, 14689),
    ('run2_subject4', 'all', 0, 8609),
    ('run2_subject4', 'all', 0, 14688),
    ('sprint1_subject2', 'all', 0, 16387),
    ('sprint1_subject4', 'all', 0, 16387),
    ('walk1_subject1', 'all', 0, 15679),
    ('walk1_subject2', 'all', 0, 15679),
    ('walk1_subject5', 'all', 0, 15679),
    ('walk2_subject1', 'all', 0, 14291),
    ('walk2_subject3', 'all', 0, 14291),
    ('walk2_subject4', 'all', 0, 14291),
    ('walk3_subject1', 'all', 0, 14795),
    ('walk3_subject2', 'all', 0, 14795),
    ('walk3_subject3', 'all', 0, 14795),
    ('walk3_subject4', 'all', 0, 14795),
    ('walk3_subject5', 'all', 0, 14795),
    ('walk4_subject1', 'all', 0, 9835),
    ('run1_subject2', 'locomotion', 0, 14269),
    ('run1_subject5', 'locomotion', 0, 14269),
    ('run2_subject1', 'locomotion', 0, 14689),
    ('run2_subject4', 'locomotion', 0, 8609),
    ('run2_subject4', 'locomotion', 11162, 14688),
    ('sprint1_subject2', 'locomotion', 0, 16387),
    ('sprint1_subject4', 'locomotion', 0, 16387),
    ('walk1_subject1', 'locomotion', 0, 15679),
    ('walk1_subject2', 'locomotion', 0, 15679),
    ('walk1_subject5', 'locomotion', 0, 15679),
    ('walk3_subject2', 'locomotion', 0, 14795),
    ('walk3_subject4', 'locomotion', 0, 6247),
    ('dance1_subject1', 'dance', 0, 7889),
    ('dance1_subject2', 'dance', 0, 7889),
    ('dance1_subject3', 'dance', 0, 7889),
    ('dance2_subject1', 'dance', 0, 13541),
    ('dance2_subject2', 'dance', 0, 13541),
    ('dance2_subject3', 'dance', 0, 13541),
    ('dance2_subject4', 'dance', 0, 13541),
    ('dance2_subject5', 'dance', 0, 13541),
    ('walk1_subject1', 'style1', 100, 300), # to delete; for debug
    ('walk1_subject1', 'style1', 400, 600), # to delete; for debug
    ('walk1_subject1', 'style1', 700, 15679), # to delete; for debug
    ('walk1_subject1', 'style2', 1200, 1500), # to delete; for debug
    ('walk1_subject1', 'style2', 1500, 1700), # to delete; for debug
    ('walk1_subject2', 'style1', 1700, 15679), # to delete; for debug
]


def generate_database(dir: Path):
    """
    Walk files under 'dir/bvh/**.bvh', process bvh data (with mirrored) and pack to a NPZ.
    Use tags_data to filter the bvh files.
    dir should look like:
    dir/
        bvh/ ... original bvh files
            file1.bvh
        database.npz
    """
    dir = Path(dir)
    input_dir = dir / "bvh"
    all_tags = ['all']

    bvh_paths = [
        input_dir / f"{range_name}.bvh"
        for range_name, tag, range_start, range_end in tags_data
        if tag in all_tags
    ]
    
    # Remove duplicates (same file can appear multiple times with different ranges)
    bvh_paths = list(dict.fromkeys(bvh_paths))
    
    if not bvh_paths: 
        raise FileNotFoundError(f"No BVH files found in {input_dir} for tags: {all_tags}")

    # initialize collectors
    bone_positions = []
    bone_velocities = []
    bone_rotations = []
    bone_angular_velocities = []
    bone_parents = []
    bone_names = []
    
    range_starts = []
    range_stops = []
    range_mirror = []
    range_names = []

    contact_states = []
    
    # Tag mapping collectors - maps each CSV tag row to X indices
    tag_range_starts = []
    tag_range_stops = []
    tag_range_names = []
    tag_tags = []
    tag_mirror = []

    for p in tqdm(bvh_paths, desc="Processing BVH files"):
        for mirror in [False, True]:
            tqdm.write(f"Processing: {p.name} {'mirrored' if mirror else 'original'}")
            range_name = p.stem  # Get filename without extension
            bvh_data = bvh.load(p.as_posix())
            positions = bvh_data['positions']
            rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order'])).astype(np.float32)
            
            # convert from cm to m
            positions *= 0.01
            
            nframes = positions.shape[0]
            nbones = positions.shape[1]
            
            """ Mirroring """
            
            if mirror:
            
                mirror_bones = []
                for ni, n in enumerate(bvh_data['names']):
                    if 'Right' in n and n.replace('Right', 'Left') in bvh_data['names']:
                        mirror_bones.append(bvh_data['names'].index(n.replace('Right', 'Left')))
                    elif 'Left' in n and n.replace('Left', 'Right') in bvh_data['names']:
                        mirror_bones.append(bvh_data['names'].index(n.replace('Left', 'Right')))
                    else:
                        mirror_bones.append(ni)
                mirror_bones = np.array(mirror_bones)
                global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])
                global_positions = np.array([-1, 1, 1]) * global_positions[:,mirror_bones]
                global_rotations = np.array([1, 1, -1, -1]) * global_rotations[:,mirror_bones]
                rotations, positions = quat.ik(global_rotations, global_positions, bvh_data['parents'])

            """ Extract Simulation Bone """
            
            # First compute world space positions/rotations
            global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])
            
            # Specify joints to use for simulation bone 
            sim_position_joint = bvh_data['names'].index("Spine2")
            sim_rotation_joint = bvh_data['names'].index("Hips")
            
            # Position comes from spine joint
            sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint:sim_position_joint+1]
            sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')
            
            # Direction comes from projected hip forward direction
            sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1], np.array([0.0, 0.0, 1.0]))

            # We need to re-normalize the direction after both projection and smoothing
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
            sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])
            
            # Extract rotation from direction
            sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

            # Transform first joints to be local to sim and append sim as root bone
            positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
            rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])
            
            positions = np.concatenate([sim_position, positions], axis=1)
            rotations = np.concatenate([sim_rotation, rotations], axis=1)
            
            bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])
            
            bone_names = ['Simulation'] + bvh_data['names']
            
            """ Compute Velocities """
            
            # Compute velocities via central difference
            velocities = np.empty_like(positions)
            velocities[1:-1] = (
                0.5 * (positions[2:  ] - positions[1:-1]) * 60.0 +
                0.5 * (positions[1:-1] - positions[ :-2]) * 60.0)
            velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
            velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
            
            # Same for angular velocities
            angular_velocities = np.zeros_like(positions)
            angular_velocities[1:-1] = (
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * 60.0 +
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * 60.0)
            angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
            angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

            """ Compute Contact Data """ 

            global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
                rotations, 
                positions, 
                velocities,
                angular_velocities,
                bone_parents)
            
            contact_velocity_threshold = 0.15
            
            contact_velocity = np.sqrt(np.sum(global_velocities[:,np.array([
                bone_names.index("LeftToeBase"), 
                bone_names.index("RightToeBase")])]**2, axis=-1))
            
            # Contacts are given for when contact bones are below velocity threshold
            contacts = contact_velocity < contact_velocity_threshold
            
            # Median filter here acts as a kind of "majority vote", and removes
            # small regions  where contact is either active or inactive
            for ci in range(contacts.shape[1]):
            
                contacts[:,ci] = ndimage.median_filter(
                    contacts[:,ci], 
                    size=6, 
                    mode='nearest')
            
            """ Append to Database """
            
            bone_positions.append(positions)
            bone_velocities.append(velocities)
            bone_rotations.append(rotations)
            bone_angular_velocities.append(angular_velocities)
            
            offset = 0 if len(range_starts) == 0 else range_stops[-1] 

            range_starts.append(offset)
            range_stops.append(offset + len(positions))
            range_names.append(range_name)
            range_mirror.append(mirror)

            contact_states.append(contacts)
            
            # Add tag mappings for all entries in tags_data matching this range_name
            for tag_range_name, tag, tag_start_in_bvh, tag_stop_in_bvh in tags_data:
                if tag_range_name == range_name:
                    # Map BVH frame indices to X indices (accounting for offset)
                    tag_start_in_x = offset + tag_start_in_bvh
                    tag_stop_in_x = offset + tag_stop_in_bvh
                    
                    tag_range_starts.append(tag_start_in_x)
                    tag_range_stops.append(tag_stop_in_x)
                    tag_range_names.append(tag_range_name)
                    tag_tags.append(tag)
                    tag_mirror.append(mirror)
            
            
    """ Concatenate Data """
        
    bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
    bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
    bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
    bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
    bone_parents = bone_parents.astype(np.int32)

    range_starts = np.array(range_starts).astype(np.int32)
    range_stops = np.array(range_stops).astype(np.int32)
    range_mirror = np.array(range_mirror).astype(np.bool)

    contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)
    
    # Convert tag mapping lists to arrays
    tag_range_starts = np.array(tag_range_starts).astype(np.int32)
    tag_range_stops = np.array(tag_range_stops).astype(np.int32)
    tag_range_names = np.array(tag_range_names, dtype=object)
    tag_tags = np.array(tag_tags, dtype=object)
    tag_mirror = np.array(tag_mirror).astype(np.bool)

    """ Save to NPZ """
    np.savez(dir / "database.npz",
                positions=bone_positions,
                velocities=bone_velocities,
                rotations=bone_rotations,
                angular_velocities=bone_angular_velocities,
                parents=bone_parents,
                names=bone_names,
                range_starts=range_starts,
                range_stops=range_stops,
                range_mirror=range_mirror,
                range_names=range_names,
                contacts=contact_states,
                tag_range_starts=tag_range_starts,
                tag_range_stops=tag_range_stops,
                tag_range_names=tag_range_names,
                tag_tags=tag_tags,
                tag_mirror=tag_mirror)

def intersect_tagged_ranges(
    tag_range_starts: np.ndarray,
    tag_range_stops: np.ndarray,
    tag_tags: np.ndarray,
    tags: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
        # Single tag: returns all style1 ranges
        (starts, stops) = intersect_tagged_ranges(starts, stops, tags, ['style1'])
        
        # Multiple tags: returns intersection
        (starts, stops) = intersect_tagged_ranges(starts, stops, tags, ['locomotion', 'style1'])
    """
    if not tags:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    if len(tags) == 1:
        mask = tag_tags == tags[0]
        return tag_range_starts[mask].copy(), tag_range_stops[mask].copy()
    
    # for each range of the first tag, find all overlapping ranges from other tags
    result_starts, result_stops = [], []
    
    # use the tag with fewest ranges as the base to minimize comparisons
    tag_ranges = {}
    for tag in tags:
        mask = tag_tags == tag
        tag_ranges[tag] = list(zip(tag_range_starts[mask], tag_range_stops[mask]))
    
    base_tag = min(tags, key=lambda t: len(tag_ranges[t]))
    other_tags = [t for t in tags if t != base_tag]
    
    # for each range of the base tag
    for base_start, base_stop in tag_ranges[base_tag]:
        # recursively intersect with other tags
        candidates = [(int(base_start), int(base_stop))]
        
        for other_tag in other_tags:
            new_candidates = []
            for cand_start, cand_stop in candidates:
                # find all ranges of other_tag that overlap with this candidate
                for other_start, other_stop in tag_ranges[other_tag]:
                    overlap_start = max(cand_start, int(other_start))
                    overlap_stop = min(cand_stop, int(other_stop))
                    
                    if overlap_start < overlap_stop:
                        new_candidates.append((overlap_start, overlap_stop))
            
            candidates = new_candidates
            if not candidates:
                break  # No overlaps found, skip this base range
        
        # add all valid intersections
        for start, stop in candidates:
            result_starts.append(start)
            result_stops.append(stop)
    
    return np.array(result_starts, dtype=np.int32), np.array(result_stops, dtype=np.int32)

# ----------------------------
# Network definitions moved to networks.py
# ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Training...")
    ap.add_argument("--dataset", type=str, default="lafan1-resolved")
    ap.add_argument("--niterations", type=int, default=500000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--pose_noise", type=float, default=0.5, help="Amount of noise to apply to previous poses")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", type=str, default='cuda', help="Device")
    ap.add_argument("--expr_name", type=str, default="experiment", help="Experiment name")
    # ap.add_argument("--window_length", type=int, default=1, help="Window length for prediction")
    args = ap.parse_args()
    print(f"Using dataset: {args.dataset}, niterations: {args.niterations}, batch size: {args.batch_size}, learning rate: {args.learning_rate}")

    RUN_NAME = time.strftime("%Y%m%d-%H%M") + f"_{args.expr_name}"
    run_dir = Path("runs") / RUN_NAME
    writer = SummaryWriter(log_dir=run_dir.as_posix())
    cfg = vars(args).copy()
    cfg["run_name"] = RUN_NAME
    writer.add_text("config/args", "<pre>"+json.dumps(cfg, indent=2)+"</pre>", global_step=0)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir = Path(__file__).resolve().parent / "outputs" / RUN_NAME
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir / "flowmatching_bvh").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir / "autoencoder_bvh").mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # download dataset if not present
    # ----------------------------
    def ensure_lafan1(out_dir: Path, url: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "lafan1-resolved-bvh.zip"
        if out_dir.exists() and any(out_dir.rglob("*/*.bvh")):
            return out_dir
        if not zip_path.exists():
            print(f"[downloading] {url} -> {zip_path}")
            urllib.request.urlretrieve(url, zip_path.as_posix())
        print(f"[extracting] {zip_path} -> {out_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        return out_dir

    DATA_ROOT = ensure_lafan1(
        out_dir=Path(__file__).resolve().parent / "data" / "lafan1_resolved",
        url="https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/bvh.zip",
    )

    # ----------------------------
    # process dataset similar to learned motion matching:
    # https://github.com/orangeduck/Motion-Matching/blob/main/resources/generate_database.py
    # https://github.com/orangeduck/lafan1-resolved/blob/main/mirror.py#L336
    # ----------------------------
    database_path = DATA_ROOT / "database.npz"
    if not database_path.exists():
        generate_database(dir=DATA_ROOT)
    database = np.load(database_path, allow_pickle=True)
    names = database['names']
    parents = database['parents']
    contacts = database['contacts'].astype(np.float32)
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    range_names = database['range_names']
    range_lens = range_stops - range_starts
    tag_range_starts = database['tag_range_starts']
    tag_range_stops = database['tag_range_stops']
    tag_tags = database['tag_tags']
    Xpos = database['positions'].astype(np.float32)
    Xrot = database['rotations'].astype(np.float32)
    Xvel = database['velocities'].astype(np.float32)
    Xang = database['angular_velocities'].astype(np.float32)
    nframes = Xpos.shape[0]
    nbones = Xpos.shape[1]
    print(f"Database contains {nframes} frames, {nbones} bones.")
    
    pose_data = {
        'Xpos': Xpos,
        'Xrot': Xrot,
        'Xvel': Xvel,
        'Xang': Xang,
        'range_starts': range_starts,
        'range_stops': range_stops,
        'range_lens': range_lens,
        'range_names': range_names,
        'tag_range_starts': database['tag_range_starts'],
        'tag_range_stops': database['tag_range_stops'],
        'tag_tags': database['tag_tags'],
    }
    
    xnpz_path = DATA_ROOT / "X.npz"
    
    if xnpz_path.exists():
        
        print(f"Loaded preprocessed X.npz from {xnpz_path}")
        
        data = np.load(xnpz_path, allow_pickle=True)
        X = data['X']
        Xref_pos = data['Xref_pos']
        Xoffset = data['Xoffset']
        Xscale = data['Xscale']
        Xdist = torch.as_tensor(data['Xdist'], device=device)
        Xwei = torch.as_tensor(data['Xwei'], device=device)
        
    else:
        
        print(f"Preprocessing data to {xnpz_path}")
        
        Xrvel = quat.inv_mul_vec(Xrot[:,0], Xvel[:,0]) # local root linear velocity
        Xrang = quat.inv_mul_vec(Xrot[:,0], Xang[:,0]) # local root angular velocity
        Xtxy = quat.to_xform_xy(Xrot).reshape(nframes, nbones, 6) # local rotations 2-axis
        Xatr = contacts # extra attributes (contact states)
        
        # Reference pose that will be used for getting local joint offsets in reconstruction
        Xref_pos = Xpos.mean(axis=0)
        
        # Construct pose vectors
        X = np.concatenate([
            Xrvel.reshape([nframes, 3]),
            Xrang.reshape([nframes, 3]),
            Xpos[:,1].reshape([nframes, 3]),
            Xtxy[:,1:].reshape([nframes, (nbones - 1) * 6]),
            Xvel[:,1].reshape([nframes, 3]),
            Xang[:,1:].reshape([nframes, (nbones - 1) * 3]),
            Xatr.reshape([nframes, 2])
        ], axis=-1)
        
        # Offset used to center the data
        Xoffset = X.mean(axis=0)
        
        # Scaling used to normalize the data
        Xscale = np.concatenate([
            Xrvel.std(axis=0).mean().repeat(3),
            Xrang.std(axis=0).mean().repeat(3),
            Xpos[:,1].std(axis=0).mean().repeat(3),
            Xtxy[:,1:].std(axis=0).mean().repeat((nbones - 1) * 6),
            Xvel[:,1].std(axis=0).mean().repeat(3),
            Xang[:,1:].std(axis=0).mean().repeat((nbones - 1) * 3),
            Xatr.std(axis=0).mean().repeat(2),
        ], axis=-1)
        
        # Get loss weights based on error propagation
        Xweights = np.array([weights_mesh[n] for n in names], dtype=np.float32)
        
        # Create pose vector weights
        Xwei = np.concatenate([
            np.ones([3]),
            np.ones([3]),
            np.ones([3]),
            Xweights[1:].repeat(6, axis=0) * (nbones - 1),
            np.ones([3]),
            Xweights[1:].repeat(3, axis=0) * (nbones - 1),
            np.ones([2]),        
        ], axis=-1)
        
        # Normalize pose vectors
        X = (X - Xoffset) / Xscale
        
        # Compute the std on each axis of the normalized pose vectors
        Xdist = X.std(axis=0)
        
        # Save the data
        np.savez(xnpz_path, 
            X=X, 
            Xoffset=Xoffset,
            Xscale=Xscale,
            Xref_pos=Xref_pos,
            Xwei=Xwei,
            Xdist=Xdist,
            names=names, 
            parents=parents)

        Xwei = torch.as_tensor(Xwei, device=device)
        Xdist = torch.as_tensor(Xdist, device=device)

    def export_pose_vector_to_bvh(filename, Xpose):
        
        # Un-normalize pose vector
        Xpose = Xpose * Xscale + Xoffset
        
        # Extract pose components
        Xrvel = Xpose[:,0:3]
        Xrang = Xpose[:,3:6]
        Xhip = Xpose[:,6:9]
        Xrot = quat.from_xform_xy(Xpose[:,9:9+(nbones-1)*6].reshape([len(Xpose), nbones - 1, 3, 2]))
        
        # Integrate root from local root velocities
        Xroot_pos = np.zeros([len(Xpose), 3])
        Xroot_rot = np.array([1,0,0,0]) * np.ones([len(Xpose), 4])
        for i in range(1, len(Xpose)):
            Xroot_vel = quat.mul_vec(Xroot_rot[i-1], Xrvel[i])
            Xroot_ang = quat.mul_vec(Xroot_rot[i-1], Xrang[i])
            Xroot_pos[i] = (Xroot_vel / 60.0) + Xroot_pos[i-1]
            Xroot_rot[i] = quat.mul(quat.from_scaled_angle_axis(Xroot_ang / 60.0), Xroot_rot[i-1])
        
        # Concatenate integrated root and also reference pose for joint translations
        Xpos = np.concatenate([Xroot_pos[:,None], Xhip[:,None], Xref_pos[2:][None].repeat(len(Xpose), axis=0)], axis=1)
        Xrot = np.concatenate([Xroot_rot[:,None], Xrot], axis=1)
        
        # Save to bvh
        bvh.save(filename, {
            'positions': Xpos,
            'rotations': np.degrees(quat.to_euler(Xrot)),
            'offsets': Xpos[0],
            'parents': parents,
            'names': names,
            'order': 'zyx'},
            save_positions=True)
    
    # Create windows of pairs of two-frames. This will be used to train the auto-encoder
    
    pose_window_indices = []
    for ri in range(len(range_starts)):
        if range_lens[ri] >= 2:
            for fi in range(range_lens[ri] - 1):
                pose_window_indices.append(range_starts[ri] + fi + np.arange(2))
    pose_window_indices = np.array(pose_window_indices, dtype=np.int64)
    
    # ----------------------------
    # build auto-encoder networks
    # ----------------------------

    # build encoder and decoder
    encoder_network = networks.MLP(inp=X.shape[1], out=256, hidden=512, depth=2)
    encoder_network.to(device)
    encoder_network.train()

    decoder_network = networks.MLP(inp=256, out=X.shape[1], hidden=512, depth=2)
    decoder_network.to(device)
    decoder_network.train()

    # ----------------------------
    # training auto-encoder
    # ----------------------------
    
    autoencoder_path = DATA_ROOT / "autoencoder.ptz"
    
    if os.path.exists(autoencoder_path):
        
        print(f"Loaded pre-trained auto-encoder from {autoencoder_path}")

        autoencoder_data = torch.load(autoencoder_path, weights_only=True, map_location=device)
        encoder_network.load_state_dict(autoencoder_data['encoder'])
        decoder_network.load_state_dict(autoencoder_data['decoder'])
        
    else:
        
        optimizer = torch.optim.AdamW(
            list(encoder_network.parameters()) + list(decoder_network.parameters()),
            lr=args.learning_rate,
            amsgrad=True)
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda i: 1.0 - float(i) / args.niterations)

        pbar = tqdm(range(args.niterations), desc="Training AutoEncoder", dynamic_ncols=True, leave=True)
        
        for step in pbar:
            
            # Sample batches from the pose sindows
            batch_indices = pose_window_indices[np.random.randint(0, len(pose_window_indices), size=[args.batch_size])]
            
            # Grab pairs of frames and auto-encode
            Xgnd = torch.as_tensor(X[batch_indices], device=device)
            Z = encoder_network(Xgnd)
            Xrec = Xdist * decoder_network(Z)
            
            # Weight losses to given them roughly the same magnitude at the start of training
            loss_pos = 3.0 * torch.mean(Xwei * torch.abs(Xgnd - Xrec))
            loss_vel = 15.0 * torch.mean(Xwei * torch.abs((Xgnd[:,1] - Xgnd[:,0]) - (Xrec[:,1] - Xrec[:,0])))
            loss_reg = 0.01 * torch.mean(torch.abs(Z))
            loss_dreg = 0.1 * torch.mean(torch.abs(Z[:,1] - Z[:,0]))
            loss = loss_pos + loss_vel + loss_reg + loss_dreg
            
            # Update the networks
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if (step + 1) % 10000 == 0 or (step + 1) == args.niterations:
                
                # Save Networks
                
                torch.save({
                    'step': step,
                    'encoder_model_state_dict': encoder_network.state_dict(),
                    'decoder_model_state_dict': decoder_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, args.output_dir / "models" / f"autoencoder_step_{step + 1}.pth")
                
                torch.save({
                    'encoder': encoder_network.state_dict(),
                    'decoder': decoder_network.state_dict(),
                }, autoencoder_path)
                
                # Validate Results
                
                valid_batch_size = 8
                valid_nframes = 120
                start_frames = np.random.randint(0, len(X) - valid_nframes, size=[valid_batch_size])
                
                Xgnd = np.concatenate([X[s:s+valid_nframes][None] for s in start_frames], axis=0)
                
                with torch.no_grad():
                    Xval = (Xdist * decoder_network(encoder_network(torch.as_tensor(Xgnd, device=device)))).cpu().numpy()
                
                for vi in range(valid_batch_size):
                    try:
                        export_pose_vector_to_bvh(args.output_dir / "autoencoder_bvh" / ('%08i_gnd_%i.bvh' % (step + 1, vi)), Xgnd[vi])
                        export_pose_vector_to_bvh(args.output_dir / "autoencoder_bvh" / ('%08i_val_%i.bvh' % (step + 1, vi)), Xval[vi])
                    except Exception as e:
                        print(f"Error exporting BVH for validation batch {vi}: {e}")
                    
    # ----------------------------
    # loading encoded data
    # ----------------------------

    znpz_path = DATA_ROOT / "Z.npz"
    
    if znpz_path.exists():
        
        print(f"Loaded preprocessed Z.npz from {znpz_path}")
        
        data = np.load(znpz_path, allow_pickle=True)
        Z = data['Z']
        Zoffset = data['Zoffset']
        Zscale = data['Zscale']
        Zdist = torch.as_tensor(data['Zdist'], device=device)
        Zmin = torch.as_tensor(data['Zmin'], device=device)
        Zmax = torch.as_tensor(data['Zmax'], device=device)

    else:
        
        # Encode all pose data using the encoder
        with torch.no_grad():
            Z = encoder_network(torch.as_tensor(X, device=device)).cpu().numpy()
            
        # Normalize encoded pose vectors
        Zoffset = Z.mean(axis=0)
        Zscale = np.repeat(Z.std(axis=0).mean(), Z.shape[1])
        Z = (Z - Zoffset) / Zscale
        
        # Compute the std, min, and max of the encoded pose vectors
        Zdist = Z.std(axis=0)
        Zmin, Zmax = Z.min(axis=0), Z.max(axis=0)
               
        # Save encoded pose vectors
        np.savez(znpz_path, 
            Z=Z,
            Zoffset=Zoffset,
            Zscale=Zscale,
            Zdist=Zdist,
            Zmin=Zmin,
            Zmax=Zmax)

        Zdist = torch.as_tensor(Zdist, device=device)
        Zmin = torch.as_tensor(Zmin, device=device)
        Zmax = torch.as_tensor(Zmax, device=device)

    # ----------------------------
    # build flow-matching networks
    # ----------------------------

    # build control encoder
    control_encoder = control_encoder.UberControlEncoder()
    # control_encoder = control_encoder.NullControlEncoder()
    control_encoder_network = control_encoder.root
    control_encoder_network.train()
    
    # Create encoder-specific directory for saving models and controls
    control_encoder_name = control_encoder.__class__.__name__
    control_encoder_dir = DATA_ROOT / control_encoder_name
    control_encoder_dir.mkdir(exist_ok=True)
    print(f"Using control-encoder-specific directory: {control_encoder_dir}")

    # build flow model 
    denoiser_network = networks.SkipCatMLP(inp=(Z.shape[1]*2 + control_encoder.output_size() + 1), out=Z.shape[1], hidden=1024, depth=10)
    denoiser_network.to(device)
    denoiser_network.train()
    
    # inference function
    @torch.no_grad()
    def inference(Zprev, Vnext, S = 4):
        
        # Encode control vector (Vnext is already the control vector, not raw input)
        Cnext = control_encoder_network(Vnext).to(device)
        
        # Sample z from initial distribution
        Znext = Zdist * torch.randn_like(Zprev, device=device)
        
        # NOTE: here there is a difference from the paper. Rather than sampling from the uniform Gaussian
        # distribution we found that you get better results if you sample from a Gaussian distribution
        # that is fitted to the encoded pose vectors. Presumably this is because the network does not 
        # have to transport samples as far since they are already closer to the target distribution.
        
        # Integrate velocity
        for s in range(S):
            t = torch.full([Zprev.shape[0], 1], s / S, device=device)
            Znext = Znext + (1 / S) * Zdist * denoiser_network(torch.cat([Znext, Zprev, Cnext, t], dim=1))
            
            # NOTE: here is another difference to the paper. We scale the output of the denoiser network
            # by the std of the encoded pose vectors. This acts as a kind of "denormalization" of the transport
            # velocity and works well in conjunction with the change noted above.
            
        return Znext

    # ----------------------------
    # training flow-matching
    # ----------------------------
    
    print(f"Constructing all training controls")

    # Generate all pairs of pose indices I and control objects V
    I, V = control_encoder.training_controls(pose_data)
    
    # Use encoder-specific controller path
    control_encoder_path = control_encoder_dir / "controller.ptz"
    
    if os.path.exists(control_encoder_path):
        
        print(f"Loaded pre-trained controller from {control_encoder_path}")

        controller_data = torch.load(control_encoder_path, weights_only=True, map_location=device)
        control_encoder_network.load_state_dict(controller_data['control_encoder'])
        denoiser_network.load_state_dict(controller_data['denoiser'])
        
    else:
        
        optimizer = torch.optim.AdamW(
            list(control_encoder_network.parameters()) + list(denoiser_network.parameters()),
            lr=args.learning_rate,
            amsgrad=True)
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda i: 1.0 - float(i) / args.niterations)

        pbar = tqdm(range(args.niterations), desc="Training Flow Matching", dynamic_ncols=True, leave=True)
        
        for step in pbar:
            
            # Sample random batch indices
            batch_indices = np.random.randint(0, len(I), size=[args.batch_size])
            
            # Sample pose indices and corresponding control vectors
            Zprev = torch.as_tensor(Z[I[batch_indices]-1], device=device)
            Znext = torch.as_tensor(Z[I[batch_indices]-0], device=device)
            Vnext = [V[bi] for bi in batch_indices]

            # Sample from Gaussian distribution.
            # NOTE: as described in the `inference` function this is different to the paper.
            Ztil = Zdist * torch.randn_like(Znext, device=device)
            
            # Sample uniform time value t
            t = torch.rand([args.batch_size,1], device=device)
            
            # Add noise to previous pose
            Zprev = Zprev + Zdist * args.pose_noise * torch.rand([args.batch_size, 1], device=device) * torch.randn_like(Zprev)
            
            # Compute linear interpolation
            Zbar = (1 - t) * Ztil + t * Znext
            
            # Compute loss
            Cnext = control_encoder_network(Vnext).to(device)
            Zdir = Zdist * denoiser_network(torch.cat([Zbar, Zprev, Cnext, t], dim=1))
            loss = torch.mean(torch.square(Zdir - (Znext - Ztil)))
            
            # Update network parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if (step + 1) % 10000 == 0 or (step + 1) == args.niterations:
                
                # Save Networks
                
                torch.save({
                    'step': step,
                    'model_state_dict': denoiser_network.state_dict(),
                    'control_encoder_state_dict': control_encoder_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, args.output_dir / "models" / f"model_step_{step + 1}.pth")
                
                torch.save({
                    'control_encoder': control_encoder_network.state_dict(),
                    'denoiser': denoiser_network.state_dict(),
                }, control_encoder_path)
                
    # Generate Test Data
    
    with torch.no_grad():
        
        test_n_frames = 600
        test_start_frame = np.random.randint(0, len(Z))
        Ztest = [torch.as_tensor(Z[test_start_frame][None], device=device)]
        for _ in range(test_n_frames - 1):
            # TODO: Does this need updating?
            from control_encoder import NullControlEncoder
            from gameplay_input import GameplayInput
            zero_control = NullControlEncoder().runtime_controls(gameplay_input=GameplayInput())
            Zpred = inference(Ztest[-1], [zero_control])
            Ztest.append(Zpred.clip(Zmin, Zmax))
            
        Ztest = torch.cat([Zp for Zp in Ztest], dim=0).cpu().numpy()
        Xtest = (Xdist * decoder_network(torch.as_tensor(Ztest * Zscale + Zoffset, device=device))).cpu().numpy()
    
    test_bvh_path = args.output_dir / "flowmatching_bvh" / "test.bvh"
    export_pose_vector_to_bvh(test_bvh_path, Xtest)
