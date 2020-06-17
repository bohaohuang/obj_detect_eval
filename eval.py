"""

"""


# Built-in

# Libs

# Own modules
import ode_utils

# Settings
IMG_FILE_NAME = '/hdd6/data/xview/xviewtest_img_px23whr3_seed17_m4_rc1_easy-bh.txt'
LBL_FILE_NAME = '/hdd6/data/xview/xviewtest_lbl_px23whr3_seed17_m4_rc1_easy-bh.txt'
IMG_DIR = r'/hdd6/data/xview/xview_validation_set/608_1cls_val'
LBL_DIR = r'/hdd6/data/xview/xview_validation_set/1_cls_xcycwh_px23whr3_val_m4_rc1_easy_seed17'
PRED_FILE_NAME = r'/hdd6/data/xview/results_syn_mixed_model4_v8_219.json'


def main():
    imgs, lbls, file_ids = ode_utils.load_data(IMG_FILE_NAME, LBL_FILE_NAME, IMG_DIR, LBL_DIR)
    preds = ode_utils.load_pred(PRED_FILE_NAME)

    for img, lbl_id in zip(imgs, file_ids):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for lbl in lbls[lbl_id]:
            if lbl[1] == 0:
                ode_utils.overlay_bbox(ax, lbl[2:], edgecolor='b')
            else:
                ode_utils.overlay_bbox(ax, lbl[2:], edgecolor='g')
        if lbl_id in preds:
            for pred in preds[lbl_id][1]:
                print(pred)
                ode_utils.overlay_bbox(ax, pred, edgecolor='r')
        plt.show()


if __name__ == '__main__':
    main()
