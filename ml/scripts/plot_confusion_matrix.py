
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "ml" / "test_evaluation.json"
OUTPUT_PATH = ROOT / "ml" / "test_confusion_matrix.png"


def main():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}"
        )

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    matrix = np.array(results["confusion_matrix"])
    class_names = results["class_names"]

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=35,
        colorbar=False,
    )

    ax.set_title(
        f"AgriVision AI - Test Confusion Matrix\n"
        f"Accuracy: {results['test_accuracy']:.2%}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Test confusion matrix saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
