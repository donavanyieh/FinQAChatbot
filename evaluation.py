"""
Calculate P@1, R@1, and F1@1 with predictions and ground truth labels
"""

def metric_calculation(predictions, labels):
    """
    Parameters:
        predictions: List of index of predictions
        labels: 2d list of ground truth integer labels index

    Returns:
        Dictionary where {"P@1": P@1, "R@1": R@1, "F1@1": F1@1}
    """

    sum_recall_1 = 0
    sum_precision_1 = 0
    # make sure that predictions and labels lists have the same length
    if len(predictions) != len(labels):
        return "Predictions and labels do not match."

    for i in range(len(predictions)):
        actual_label = labels[i]
        if predictions[i] in actual_label:
            sum_precision_1 += 1
            sum_recall_1 += 1/len(actual_label)

    precision_1 = sum_precision_1/len(predictions)
    recall_1 = sum_recall_1/len(predictions)
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

    return {"P@1": precision_1, "R@1": recall_1, "F1@1": f1_1}

