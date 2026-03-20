from sample_human import sample_human
import numpy as np
import opensim as osm

def perturbate_motion(
    sampled_height: float,
    reference_height: float = 1.70
) -> osm.TimeSeriesTable:

    rng    = np.random.default_rng()
    table  = osm.TimeSeriesTable("normal.mot")
    times  = np.array(list(table.getIndependentColumn()))
    labels = list(table.getColumnLabels())

    height_factor = sampled_height / reference_height
    speed_base    = np.sqrt(height_factor)
    speed_factor  = np.clip(speed_base * rng.uniform(0.9, 1.1), 0.7, 1.3)
    stride_factor = np.clip(
        0.6 * speed_factor + 0.4 * height_factor + rng.uniform(-0.03, 0.03),
        0.7, 1.3
    )
    asym_r = rng.uniform(0.97, 1.03)
    asym_l = rng.uniform(0.97, 1.03)

    scale_rules = {
    #x motion of pelvis is exactly proportional to the stride length
    "pelvis_tx":        stride_factor,

    #pelvis height is exactly proportional to the height 
    "pelvis_ty":        height_factor,

    #hip flexion is proportional to the stride length (+ asymmetry)
    "hip_flexion_r":    stride_factor * asym_r,
    "hip_flexion_l":    stride_factor * asym_l,

    #knee angle is proprtional to the speed (+ asymmetry)
    "knee_angle_r":     speed_factor * asym_r,
    "knee_angle_l":     speed_factor * asym_l,

    #ankle angle are proportional to the speed
    "ankle_angle_r":    speed_factor,
    "ankle_angle_l":    speed_factor,

    #we found no real results so just asymmetry as a factor
    "hip_adduction_r":  asym_r,
    "hip_adduction_l":  asym_l,
    "hip_rotation_r":   asym_r,
    "hip_rotation_l":   asym_l,

    #we found no real results and these parameters are influenced by asymmetry between sides
    "pelvis_tilt":      1.0,
    "pelvis_list":      1.0,
    "pelvis_tz":        1.0,
    "pelvis_rotation":  1.0,
    "lumbar_extension": 1.0,
    "lumbar_bending":   1.0,
    "lumbar_rotation":  1.0,
}

    fixed_times = np.linspace(times[0], times[-1], len(times))

    data_matrix = np.zeros((len(times), len(labels)))
    for col_idx, label in enumerate(labels):
        col       = table.getDependentColumn(label)
        vals      = np.array([col[i] for i in range(col.size())])
        vals      = vals * scale_rules.get(label, 1.0)
        if label == "pelvis_ty":
        #pelvis height is constant, otherwise the model "wiggles" up and down
            vals = np.full(len(vals), 0.957 * height_factor)
        else:
            vals = vals * scale_rules.get(label, 1.0)
            amplitude = np.max(np.abs(vals)) + 1e-6
            vals = vals + rng.normal(0, 0.01 * amplitude, size=len(vals))
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


