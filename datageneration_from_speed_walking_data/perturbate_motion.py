from sample_human import sample_human
import numpy as np
import opensim as osm
from scipy.ndimage import gaussian_filter1d
import tqdm
import os
import random
import numpy as np
from scipy.interpolate import interp1d

def perturbate_motion(
    sampled_height: float,
    motion_path: str,
    reference_height: float = 1.70,
) -> str:
    rng    = np.random.default_rng()
    table  = osm.TimeSeriesTable(motion_path)
    times  = np.array(list(table.getIndependentColumn()))
    labels = list(table.getColumnLabels())

    height_factor = sampled_height / reference_height
    speed_base    = np.sqrt(height_factor)
    speed_factor  = np.clip(speed_base * rng.uniform(0.9, 1.1), 0.7, 1.3)
    stride_factor = np.clip(
        0.6 * speed_factor + 0.4 * height_factor + rng.uniform(-0.03, 0.03),
        0.7, 1.3
    )
    asym_r = rng.uniform(0.95, 1.05)
    asym_l = rng.uniform(0.95, 1.05)

    scale_rules = {
        "pelvis_tx":        stride_factor,
        "pelvis_ty":        height_factor,
        "hip_flexion_r":    stride_factor * asym_r,
        "hip_flexion_l":    stride_factor * asym_l,
        "knee_angle_r":     speed_factor * asym_r,
        "knee_angle_l":     speed_factor * asym_l,
        "ankle_angle_r":    speed_factor,
        "ankle_angle_l":    speed_factor,
        "hip_adduction_r":  asym_r,
        "hip_adduction_l":  asym_l,
        "hip_rotation_r":   asym_r,
        "hip_rotation_l":   asym_l,
        "pelvis_tilt":      1.0,
        "pelvis_list":      1.0,
        "pelvis_tz":        1.0,
        "pelvis_rotation":  1.0,
        "lumbar_extension": 1.0,
        "lumbar_bending":   1.0,
        "lumbar_rotation":  1.0,
    }

    #angle cut offs to avoid unrealistic movements
    angle_limits = {
        "hip_adduction_r":  (-15, 15),
        "hip_adduction_l":  (-15, 15),
        "hip_rotation_r":   (-10, 10),
        "hip_rotation_l":   (-10, 10),
        "lumbar_bending":   (-10, 10),
        "lumbar_rotation":  (-10, 10),
        "pelvis_list":      (-10, 10),
        "pelvis_rotation":  (-10, 10),
    }

    fixed_times = np.linspace(times[0], times[-1], len(times))
    data_matrix = np.zeros((len(times), len(labels)))

    for col_idx, label in enumerate(labels):
        col  = table.getDependentColumn(label)
        vals = np.array([col[i] for i in range(col.size())])
        vals = vals * scale_rules.get(label, 1.0)

        if label == "pelvis_ty":
            vals = np.full(len(vals), 0.957 * height_factor)
        else:
            vals = vals * scale_rules.get(label, 1.0)

            #smoothed noise
            amplitude = np.max(np.abs(vals)) + 1e-6
            noise     = rng.normal(0, 0.02 * amplitude, size=len(vals))
            noise     = gaussian_filter1d(noise, sigma=5)   
            vals      = vals + noise

        #apply angle cut offs
        if label in angle_limits:
            lo, hi = angle_limits[label]
            vals = np.clip(vals, lo, hi)

        data_matrix[:, col_idx] = vals

    matrix = osm.Matrix(len(times), len(labels))
    for i in range(len(times)):
        for j in range(len(labels)):
            matrix.set(i, j, data_matrix[i, j])

    new_table = osm.TimeSeriesTable(
        list(fixed_times),
        matrix,
        list(table.getColumnLabels())
    )
    new_table.addTableMetaDataString("inDegrees", "yes")
    osm.STOFileAdapter.write(new_table, "perturbed_motion.mot")
    return "perturbed_motion.mot"

#later we use interpolation on the foot and com data to get the sampling rate and times of the frames to match for training. This function does the same directly on the motion files, so we can visualize the models motion in the opensim GUI. It is only used for debugging/reality check.
def resample_mot(input_path, output_path, target_hz=50):
    table     = osm.TimeSeriesTable(input_path)
    times     = np.array(list(table.getIndependentColumn()))
    labels    = list(table.getColumnLabels())

    n_new     = int((times[-1] - times[0]) * target_hz)
    times_new = np.linspace(times[0], times[-1], n_new)

    data_old  = np.zeros((len(times), len(labels)))
    for j, label in enumerate(labels):
        col = table.getDependentColumn(label)
        data_old[:, j] = [col[i] for i in range(col.size())]

    data_new = interp1d(times, data_old, axis=0, kind='linear')(times_new)

    matrix = osm.Matrix(n_new, len(labels))
    for i in range(n_new):
        for j in range(len(labels)):
            matrix.set(i, j, float(data_new[i, j]))

    new_table = osm.TimeSeriesTable(list(times_new), matrix, labels)
    new_table.addTableMetaDataString("inDegrees", "yes")
    osm.STOFileAdapter.write(new_table, output_path)



