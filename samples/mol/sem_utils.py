# -*- coding: utf-8 -*-
import os
import re
import numpy as np

def run_mask_rcnn(image_path):
    """
    run mask rcnn on given image.
    
    Returns:
        None
    Output:
        image with name _pred.png
        numpy array contains rois and masks in _pred.npz
    """
    image = skimage.io.imread(image_path)
    results = model.detect([image], verbose=0)

    # Display results
    ax, fig = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    fig.savefig(image_path[:-4]+'_pred.png', bbox_inches='tight')
    np.savez(image_path[:-4]+'_pred', rois=r['rois'], masks=r['masks'])

def read_npz(image_path):
    """
    read mask rcnn prediction, return bbox coordinate and mask image

    Returns:
        rois: bbox array of (4, n)
        masks: image mask array of (h, w, n)
    """
    path = image_path[:-4] + '_pred.npz'
    result = np.load(path)
    return result['rois'], result['masks'], result['class_ids'], result['scores']

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)