def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    def iou(box_a, box_b):
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
        area_b = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])
        return intersection_area / (area_a + area_b - intersection_area) if (area_a + area_b - intersection_area) else 0
    def get_index_sorted_score(scores):
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return indices
    result = []

    index_sorted_score = get_index_sorted_score(scores)
    while index_sorted_score:
        current_max_index = index_sorted_score.pop(0)
        result.append(current_max_index)
        remaining_index = []
        for i in index_sorted_score:
            iou_score = iou(boxes[i], boxes[current_max_index])
            if iou_score < iou_threshold:
                remaining_index.append(i)
        index_sorted_score = remaining_index
    return result