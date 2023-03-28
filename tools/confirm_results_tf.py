import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

NUM_JOINTS: int = 14
KINEM_IDX: dict = {
    "Head": 0, "Neck": 1, "Spine1": 2, "Pelvis": 3,
    "L_Hand": 4, "L_Wrist": 5, "L_Elbow": 6, "L_Shoulder": 7,
    "R_Hand": 8, "R_Wrist": 9, "R_Elbow": 10, "R_Shoulder": 11,
    "L_Hip": 12,
    "R_Hip": 13
}
KINEM_PAR: dict = {
    "Head":"Neck", "Neck":"Neck", "Spine1":"Neck", "Pelvis":"Spine1",
    "L_Hand":"L_Wrist", "L_Wrist":"L_Elbow", "L_Elbow":"L_Shoulder", "L_Shoulder":"Neck",
    "R_Hand":"R_Wrist", "R_Wrist":"R_Elbow", "R_Elbow":"R_Shoulder", "R_Shoulder":"Neck",
    "L_Hip":"Pelvis",
    "R_Hip":"Pelvis"
}

def load(val_dir:str) -> (np.ndarray, dict, dict):
    val_images = np.load(f"{val_dir}/val_images.npy")

    gts = np.load(f"{val_dir}/gts.npy", allow_pickle=True)
    gts = gts.item()

    preds = np.load(f"{val_dir}/preds.npy", allow_pickle=True)
    preds = preds.item()

    return val_images, gts, preds

def show(val_images:np.ndarray, gts:dict, preds:dict) -> None:
    gt_hmaps = gts["heatmaps"]
    gt_joints_2d = gts["joints_2d"]
    gt_joints_3d = gts["joints_3d"]

    pred_hmaps = preds["heatmaps"]
    pred_joints_2d = preds["joints_2d"]
    pred_joints_3d = preds["joints_3d"]

    # 3D plot preparation
    plt.ion()
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection="3d")
    gt_kinem_plots = []
    pred_kinem_plots = []
    for key in KINEM_IDX.keys():
        gt_plot, = axes.plot3D((0,0), (0,0), (0,0),  marker="", linewidth=1.0, color="blue")
        gt_kinem_plots.append(gt_plot)
        pred_plot, = axes.plot3D((0,0), (0,0), (0,0),  marker="", linewidth=1.0, color="red")
        pred_kinem_plots.append(pred_plot)
    axes.set_xlabel('X')
    axes.set_xlim(-1.5, 1.5)
    axes.set_ylabel('Y')
    axes.set_ylim(-1.7, 1.3)
    axes.set_zlabel('Z')
    axes.set_zlim(-2.3, 0.7)
    axes.view_init(elev=20, azim=110)

    for val_img, gt_j2d, gt_j3d, pred_j2d, pred_j3d in zip(val_images, gt_joints_2d, gt_joints_3d, pred_joints_2d, pred_joints_3d):
        # show 3d pose liftup result
        for key in KINEM_IDX.keys():
            gt_j = gt_j3d[KINEM_IDX[key]]
            gt_par_j = gt_j3d[KINEM_IDX[KINEM_PAR[key]]]

            gt_plot = gt_kinem_plots[KINEM_IDX[key]]
            gt_plot.set_xdata(np.array([gt_j[0], gt_par_j[0]]))
            gt_plot.set_ydata(np.array([gt_j[1], gt_par_j[1]]))
            gt_plot.set_3d_properties(np.array([gt_j[2], gt_par_j[2]]))

            pred_j = pred_j3d[KINEM_IDX[key]]
            pred_par_j = pred_j3d[KINEM_IDX[KINEM_PAR[key]]]

            pred_plot = pred_kinem_plots[KINEM_IDX[key]]
            pred_plot.set_xdata(np.array([pred_j[0], pred_par_j[0]]))
            pred_plot.set_ydata(np.array([pred_j[1], pred_par_j[1]]))
            pred_plot.set_3d_properties(np.array([pred_j[2], pred_par_j[2]]))

        fig.canvas.draw()
        fig.canvas.flush_events()

        # show pose estimation result
        h_img, w_img = val_img.shape[:2]
        for gt_j, pred_j in zip(gt_j2d, pred_j2d):
            cv2.circle(val_img, (int(w_img*gt_j[0]), int(h_img*gt_j[1])), 5,(255, 0, 0), thickness=-1)
            cv2.circle(val_img, (int(w_img*pred_j[0]), int(h_img*pred_j[1])), 5,(0, 0, 255), thickness=-1)

            print(f"gt: {gt_j}, pred: {pred_j}")

        cv2.imshow('validation image', val_img)

        key = cv2.waitKey(30)
        if 27 == key: # push 'ESC'
            break

    cv2.destroyAllWindows()
    plt.close()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='show results path')
    # general
    parser.add_argument('--val_dir',
                        help='path to validation directory',
                        required=True,
                        type=str)

    args = parser.parse_args()

    val_images, gts, preds = load(args.val_dir)

    show(val_images, gts, preds)
