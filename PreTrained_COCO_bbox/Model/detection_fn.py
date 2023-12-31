import torch
import numpy as np
import tqdm


def init_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.1)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Compute AP", leave=False):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_data_config(path: str):
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    base_path = options['train'].split('/')
    options['base'] = '/'.join(base_path[0:base_path.index('coco')])
    return options


def load_classes(path: str):
    with open(path, "r") as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names


def rescale_boxes_original(prediction, rescaled_size: int, original_size: tuple):
    """Rescale bounding boxes to the original shape"""
    ow, oh = original_size
    resize_ratio = rescaled_size / max(original_size)

    if ow > oh:
        resized_w = rescaled_size
        resized_h = round(min(original_size) * resize_ratio)
        pad_x = 0
        pad_y = abs(resized_w - resized_h)
    else:
        resized_w = round(min(original_size) * resize_ratio)
        resized_h = rescaled_size
        pad_x = abs(resized_w - resized_h)
        pad_y = 0

    # Rescale bounding boxes
    prediction[:, 0] = (prediction[:, 0] - pad_x // 2) / resize_ratio
    prediction[:, 1] = (prediction[:, 1] - pad_y // 2) / resize_ratio
    prediction[:, 2] = (prediction[:, 2] - pad_x // 2) / resize_ratio
    prediction[:, 3] = (prediction[:, 3] - pad_y // 2) / resize_ratio

    for i in range(prediction.shape[0]):
        for k in range(0, 3, 2):
            if prediction[i][k] < 0:
                prediction[i][k] = 0
            elif prediction[i][k] > ow:
                prediction[i][k] = ow
        for k in range(1, 4, 2):
            if prediction[i][k] < 0:
                prediction[i][k] = 0
            elif prediction[i][k] > oh:
                prediction[i][k] = oh

    return prediction


def xywh2xyxy(x):
    # input x = [cx, cy, w, h]
    # output y = [x1, y1, x2, y2]
    #            (x1, y1): the top-left corner point
    #            (x2, y2): the bottom-right corner point
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def get_batch_statistics(outputs, targets, iou_threshold):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    for i, output in enumerate(outputs):

        if output is None:
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    # wh: width height
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # intersection 좌표 구하기
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of two bounding boxes."""
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def non_max_suppression(prediction, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        detections = image_pred[:, :5]

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, target, anchors, ignore_thres, device):
    nB = pred_boxes.size(0)  # num. of batches
    nA = pred_boxes.size(1)  # num. of anchors
    nG = pred_boxes.size(2)  # grid size

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool,
                           device=device)  # (num_boxes, num_anchors, grid size, grid size)
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool,
                            device=device)  # (num_boxes, num_anchors, grid size, grid size)

    # Each grid cell has the IOU value between each anchor box and a ground truth.
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float,
                             device=device)  # (num_boxes, num_anchors, grid size, grid size)

    #
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)

    # ------- Convert to position relative to box -------
    # target: (num. of boxes of all images in a batch, 6)
    # target[0] = [image_id, cls, cx, cy, w, h]
    # By "* nG" in the lower code line, cx, cy, w, h become relative to the grid size of a feature map
    scaled_target_boxes = nG * target[:, 2:6]  # cx, cy, w, and h
    # target_boxes: (num. of boxes, 4)
    # target_boxes[0] = [cx, cy, w, h]
    gxy = scaled_target_boxes[:, :2]
    gwh = scaled_target_boxes[:, 2:]

    # ------- Get anchors with best iou -------
    # len(anchors) = 3, the number of anchor boxes | anchor = (w, h) | g_wh: (num boxes, w, h)
    # bbox_wh_iou(anchor, g_wh) calculates the IOU between an anchor and all ground truth boxes
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # ious: (3, num boxes), 3 is the number of anchor boxes and batch_size
    # ious includes the IOU between all ground truth bboxes and all anchors
    #            object 1 g_bbox  object 2 g_bbox  object 3 g_bbox
    # anchor 1        0.9               0.7              0.5
    # anchor 2        0.8               0.5              0.93
    # anchor 3        0.7               0.8              0.8
    _, best_ious_idx = ious.max(0)  # extract the best anchor with the highest IOU
    # best_iou_wa = [0.9, 0.8, 0.93], best_anchor_ind = [0 (anchor 1), 2 (anchor 3), 1 (anchor 2)]
    # In other words, best_anchor_ind includes the matching between an anchor and an ground truth bbox
    #
    #                                      object 1 g_bbox  object 2 g_bbox  object 3 g_bbox
    # best_anchor_ind = [0, 2, 1]  means      anchor 1         anchor 3         anchor 2
    #
    # Separate target values
    # target: (num_boxes, 6)
    # target[0] = [image_id, cls, cx, cy, w, h]
    # target[:, :2]: (num_boxes, 2)
    # target[:, :2][0] = [image_id, cls]
    b, target_labels = target[:, :2].long().t()
    # b: image_id | target_labels: class_id
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # .long() makes the type of gi and gj integer.

    # Set masks
    # obj_mask: (num boxes, num anchors, grid size, grid size)
    obj_mask[b, best_ious_idx, gj, gi] = 1
    noobj_mask[b, best_ious_idx, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # ious: (num anchors, num boxes)
    # ious.t(): (num boxes, num anchors)
    # Anchors with IOU higher than the ignoring threshold
    # The IOU value of the best anchor cannot be larger than the ignoring threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    # remind the equation as
    # g_w = p_w * exp(tw) -> tw = ln(g_w / p_w), hence p_w == anchor_w
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # Compute label correctness and iou at best anchor
    # class_mask: (num_boxes, num_anchors, grid size, grid size)
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], scaled_target_boxes,
                                                    x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, obj_mask, noobj_mask, tx, ty, tw, th, tconf
