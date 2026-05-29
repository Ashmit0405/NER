import numpy as np
import evaluate
metric = evaluate.load("seqeval")

def compute_metrics(p, label_names):

    predictions, labels = p
    predictions = np.argmax(
        predictions,
        axis=2
    )
    true_predictions = [
        [
            label_names[p]
            for (p, l) in zip(pred, lab)
            if l != -100
        ]
        for pred, lab in zip(predictions, labels)
    ]

    true_labels = [
        [
            label_names[l]
            for (p, l) in zip(pred, lab)
            if l != -100
        ]
        for pred, lab in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }