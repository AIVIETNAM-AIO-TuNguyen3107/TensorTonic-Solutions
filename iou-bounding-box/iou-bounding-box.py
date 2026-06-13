def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    intersect_area = max(0, inter_x2-inter_x1)*max(0, inter_y2-inter_y1)

    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])

    iou = intersect_area / (area_a + area_b - intersect_area)
    return iou