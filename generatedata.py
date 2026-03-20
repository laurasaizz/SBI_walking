import opensim as osim
import numpy as np
from sample_human import sample_human

def leg_length(model, state):
    hip_r = model.getJointSet().get("hip_r")
    knee_r = model.getJointSet().get("walker_knee_r")
    ankle_r = model.getJointSet().get("ankle_r")

    hip_pos = hip_r.getChildFrame().getPositionInGround(state)
    knee_pos = knee_r.getChildFrame().getPositionInGround(state)
    ankle_pos = ankle_r.getChildFrame().getPositionInGround(state)

    thigh = np.linalg.norm([
        hip_pos.get(i) - knee_pos.get(i)
        for i in range(3)
    ])

    shank = np.linalg.norm([
        knee_pos.get(i) - ankle_pos.get(i)
        for i in range(3)
    ])

    return thigh + shank

def generate_single_sample(model_path, mot_path):

    model = osim.Model(model_path)
    state:osim.State = model.initSystem()

    table = osim.TimeSeriesTable(mot_path)

    processor = osim.TableProcessor(table)
    processor.append(osim.TabOpConvertDegreesToRadians())
    table = processor.process(model)

    coordSet = model.getCoordinateSet()
    labels = table.getColumnLabels()

    coord_indices = []
    coords = []

    for j in range(len(labels)):
        if coordSet.contains(labels[j]):
            coord_indices.append(j)
            coords.append(coordSet.get(labels[j]))

    bodySet = model.getBodySet()
    foot_r = bodySet.get("calcn_r")
    foot_l = bodySet.get("calcn_l")

    N = table.getNumRows()

    foot_data = np.zeros((N, 6))
    times = table.getIndependentColumn()

    leglength = None

    for i in range(N):

        state.setTime(times[i])
        row = table.getRowAtIndex(i)

        for k, coord in enumerate(coords):
            coord.setValue(state, row[coord_indices[k]], False)

        model.realizePosition(state)

        if i == 0:
            leglength = leg_length(model, state)

        pos_r = foot_r.getPositionInGround(state)
        pos_l = foot_l.getPositionInGround(state)

        foot_data[i, :] = [
            pos_r.get(0), pos_r.get(1), pos_r.get(2),
            pos_l.get(0), pos_l.get(1), pos_l.get(2)
        ]

    # remove drift
    foot_data -= foot_data[0]

    return foot_data, np.array([leglength])

def generate_dataset(
    num_samples,
    mot_path,
    output_file="dataset.npz"
):

    all_foot = []
    all_leg = []

    import tqdm
    for i in tqdm.tqdm(range(num_samples)):

        print(f"Generating sample {i+1}/{num_samples}")

        model_file = f"model_{i}.osim"

        sample_human(model_file)

        foot_data, leg_length = generate_single_sample(
            model_file, mot_path
        )

        all_foot.append(foot_data)
        all_leg.append(leg_length)

    # stack dataset
    all_foot = np.array(all_foot)   # (num_samples, N, 6)
    all_leg = np.array(all_leg)     # (num_samples, 1)

    np.savez(output_file,
             foot=all_foot,
             leg=all_leg)

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    n = 2000
    osim.Logger.setLevelString('error')
    generate_dataset(100,"normal.mot")
