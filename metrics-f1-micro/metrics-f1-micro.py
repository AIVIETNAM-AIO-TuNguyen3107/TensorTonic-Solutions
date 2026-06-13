def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp = 0
    for real, pred in zip(y_true, y_pred):
        if real == pred:
            tp +=1
    return tp / len(y_true)