"""
generatedata.py

Generates a dataset of foot positions and COM trajectories by sampling
random human models and perturbing real motion capture files.

Each sample is resampled to TARGET_HZ and trimmed to the length of the
shortest motion file in the motion_dateien folder, so all sequences have
the same shape.
"""

import opensim as osim
import numpy as np
from scipy.interpolate import interp1d
from sample_human import sample_human
from perturbate_motion import perturbate_motion
import tqdm
import os
import random

TARGET_HZ     = 50
TARGET_FRAMES = None  # set automatically from the shortest motion file


def get_target_frames(motion_dir, target_hz=TARGET_HZ):
    """Returns the number of frames in the shortest motion file at target_hz."""
    mot_files  = [os.path.join(motion_dir, f)
                  for f in os.listdir(motion_dir) if f.endswith(".mot")]
    min_frames = float("inf")
    shortest   = None

    for path in mot_files:
        try:
            table    = osim.TimeSeriesTable(path)
            times    = list(table.getIndependentColumn())
            duration = times[-1] - times[0]
            n_frames = int(duration * target_hz)
            if n_frames < min_frames:
                min_frames = n_frames
                shortest   = path
        except Exception as e:
            print(f"  Warning: could not read {path} ({e})")

    print(f"  Shortest motion file : {os.path.basename(shortest)}")
    print(f"  Target frames        : {min_frames} ({min_frames/target_hz:.2f}s at {target_hz} Hz)")
    return min_frames


def normalize_sequence(data, times, target_hz=TARGET_HZ, target_frames=None):
    """
    Resamples (T, D) data with irregular timestamps to uniform target_hz,
    then trims to target_frames.
    """
    times     = np.array(times, dtype=np.float64)
    duration  = times[-1] - times[0]
    n_uniform = max(2, int(duration * target_hz))
    t_uniform = np.linspace(times[0], times[-1], n_uniform)

    data_uniform = interp1d(times, data, axis=0, kind="linear")(t_uniform)

    n = min(n_uniform, target_frames)
    return data_uniform[:n].astype(np.float32)


def generate_single_sample(model_path, mot_path, target_frames=TARGET_FRAMES):
    """
    Runs forward kinematics on model_path driven by mot_path and returns
    foot positions and COM resampled to TARGET_HZ with target_frames frames.
    """
    model = osim.Model(model_path)
    state = model.initSystem()

    table = osim.TimeSeriesTable(mot_path)
    processor = osim.TableProcessor(table)
    try:
        if table.getTableMetaDataString("inDegrees").strip().lower() == "yes":
            processor.append(osim.TabOpConvertDegreesToRadians())
    except Exception:
        pass
    table = processor.process(model)

    coordSet = model.getCoordinateSet()
    labels   = table.getColumnLabels()
    coord_indices, coords = [], []
    for j in range(len(labels)):
        if coordSet.contains(labels[j]):
            coord_indices.append(j)
            coords.append(coordSet.get(labels[j]))

    foot_r = model.getBodySet().get("calcn_r")
    foot_l = model.getBodySet().get("calcn_l")

    N         = table.getNumRows()
    times     = list(table.getIndependentColumn())
    foot_data = np.zeros((N, 6))
    com_data  = np.zeros((N, 3))

    for i in range(N):
        state.setTime(times[i])
        row = table.getRowAtIndex(i)
        for k, coord in enumerate(coords):
            coord.setValue(state, row[coord_indices[k]], False)
        model.realizePosition(state)

        pr = foot_r.getPositionInGround(state)
        pl = foot_l.getPositionInGround(state)
        foot_data[i] = [pr.get(0), pr.get(1), pr.get(2),
                        pl.get(0), pl.get(1), pl.get(2)]
        c = model.calcMassCenterPosition(state)
        com_data[i]  = [c.get(0), c.get(1), c.get(2)]

    foot_data = normalize_sequence(foot_data, times, target_frames=target_frames)
    com_data  = normalize_sequence(com_data,  times, target_frames=target_frames)
    return foot_data, com_data


def generate_dataset(num_samples, output_file="dataset.npz"):
    motion_dir = os.path.join(os.path.dirname(__file__), "motion_dateien")
    mot_files  = [f for f in os.listdir(motion_dir) if f.endswith(".mot")]
    if not mot_files:
        raise FileNotFoundError(f"No .mot files found in {motion_dir}")
    print(f"Motion files found : {len(mot_files)}")

    target_frames = get_target_frames(motion_dir)

    all_foot, all_com, all_height = [], [], []
    pbar      = tqdm.tqdm(total=num_samples)
    attempts  = 0
    discarded = 0

    while len(all_foot) < num_samples:
        attempts += 1
        random_mot = os.path.join(motion_dir, random.choice(mot_files))
        model, height = sample_human("model.osim")
        model.printToXML("temporary_model.osim")
        mot_path  = perturbate_motion(height, random_mot)
        foot_data, com_data = generate_single_sample(
            "temporary_model.osim", mot_path, target_frames)

        # discard samples where the model starts in an implausible position
        if com_data[0, 1] < 0.85:
            discarded += 1
            continue

        all_foot.append(foot_data)
        all_com.append(com_data)
        all_height.append(height)
        pbar.update(1)

    pbar.close()
    print(f"  Discarded {discarded} of {attempts} samples ({discarded/attempts*100:.1f}%)")

    all_foot   = np.array(all_foot,   dtype=np.float32)  # (S, T, 6)
    all_com    = np.array(all_com,    dtype=np.float32)  # (S, T, 3)
    all_height = np.array(all_height, dtype=np.float32)  # (S,)

    np.savez(output_file, foot=all_foot, com=all_com, height=all_height)
    print(f"Dataset saved -> {output_file}")
    print(f"  foot shape : {all_foot.shape}")
    print(f"  COM shape  : {all_com.shape}")


if __name__ == "__main__":
    osim.Logger.setLevelString("error")
    generate_dataset(100)