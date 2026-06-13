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
        return intersection_area / (area_a + area_b - intersection_area)
    def get_index_sorted_score(scores):
        dict_scores = {i: score for i, score in enumerate(scores)}
        sorted_score = sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)
        return [index for index, score in sorted_score]
    result = []
    surpressed_index = set()

    index_sorted_score = get_index_sorted_score(scores)
    for max_index in index_sorted_score:
        if max_index in surpressed_index:
            continue
        result.append(max_index)
        for i, box in enumerate(boxes):
            if i == max_index:
                continue
            if i in surpressed_index:
                continue
            iou_score = iou(box, boxes[max_index])
            if iou_score >= iou_threshold:
                surpressed_index.add(i)
    return result