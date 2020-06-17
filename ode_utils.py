"""

"""


# Built-in
import os

# Libs
import toolman as tm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Own modules


def parse_raw(raw_line, width, height):
    class_id, xcenter, ycenter, w, h, type_id = raw_line.strip().split(' ')
    class_id, type_id = int(class_id), int(type_id)
    xcenter, ycenter, w, h = float(xcenter), float(ycenter), float(w), float(h)

    xmin, xmax = width*(xcenter-w/2), width*(xcenter+w/2)
    ymin, ymax = height*(ycenter-h/2), height*(ycenter+h/2)

    return class_id, type_id, xmin, ymin, xmax, ymax


def load_data(img_file_list, lbl_file_list, img_dir, lbl_dir):
    img_files = [os.path.basename(a.strip()).split('.')[0] for a in tm.misc_utils.load_file(img_file_list)]
    lbl_files = [os.path.basename(a.strip()).split('.')[0] for a in tm.misc_utils.load_file(lbl_file_list)]
    assert img_files == lbl_files

    imgs = [tm.misc_utils.load_file(os.path.join(img_dir, f'{a}.jpg')) for a in img_files]
    height, width = tm.misc_utils.load_file(os.path.join(img_dir, f'{img_files[0]}.jpg')).shape[:2]

    lbls = {}
    for lbl_file in lbl_files:
        lbl_raw = tm.misc_utils.load_file(os.path.join(lbl_dir, f'{lbl_file}.txt'))
        lbls[lbl_file] = [parse_raw(a, height, width) for a in lbl_raw]

    return imgs, lbls, img_files


def load_pred(pred_file):
    preds_raw = tm.misc_utils.load_file(pred_file)
    preds = {}
    for p in preds_raw:
        file_id = p['image_name'].split('.')[0]
        class_id = p['category_id']
        bbox = p['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        conf = p['score']
        if file_id not in preds:
            preds[file_id] = [[class_id], [bbox], [conf]]
        else:
            preds[file_id][0].append(class_id)
            preds[file_id][1].append(bbox)
            preds[file_id][2].append(conf)

    return preds


def overlay_bbox(ax, coords, linewidth=1, edgecolor='r'):
    xmin, ymin, xmax, ymax = coords
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=linewidth, edgecolor=edgecolor,
                             facecolor='none')
    ax.add_patch(rect)
