import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

def process_npz(npz_path, out_dir, nu=0.01):
    # Load latent embeddings and labels
    data = np.load(npz_path)
    latents_normal = data['latents_normal']
    latents_test   = data['latents_test']
    true_labels    = data['true_labels']

    print(f"Processing {os.path.basename(out_dir)}:")
    print(f"  Normal latents: {latents_normal.shape}, Test latents: {latents_test.shape}")

    # Train One-Class SVM on normal latents
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
    oc_svm.fit(latents_normal)

    # Predict and score on test set
    preds_svm = oc_svm.predict(latents_test)  # +1 = inlier, -1 = outlier
    preds = np.where(preds_svm == 1, 0, 1)    # Map to binary: 0 = normal, 1 = tumor
    scores = -oc_svm.decision_function(latents_test)

    # Metrics
    report = classification_report(true_labels, preds, target_names=["Normal", "Tumor"])
    cm = confusion_matrix(true_labels, preds)
    auc = roc_auc_score(true_labels, scores)

    # Save textual report
    txt_path = os.path.join(out_dir, 'ocsvm_report.txt')
    with open(txt_path, 'w') as f:
        f.write(f"OCSVM Report for {os.path.basename(out_dir)}\n")
        f.write(report + '\n')
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + '\n')
        f.write(f"ROC AUC (Tumor positive): {auc:.4f}\n")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(true_labels, scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Tumor Detection)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()

    # Plot score histogram
    scores_normal = scores[true_labels == 0]
    scores_tumor  = scores[true_labels == 1]
    plt.figure()
    plt.hist([scores_normal, scores_tumor], bins=50, label=["Normal", "Tumor"], alpha=0.7)
    plt.xlabel('One-Class SVM decision function score')
    plt.ylabel('Count')
    plt.title('Distribution of OCSVM Scores by True Class')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'score_histogram.png'))
    plt.close()

    print(f"  Done. ROC AUC = {auc:.4f}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch OCSVM analysis over latent directories'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Root folder containing latent_<dim> subdirectories'
    )
    parser.add_argument(
        '--nu', type=float, default=0.01,
        help='OCSVM nu parameter'
    )
    args = parser.parse_args()

    root = args.output_dir
    if not os.path.isdir(root):
        raise ValueError(f"Provided output_dir does not exist or is not a directory: {root}")

    # Loop over each latent_* folder
    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue
        npz_file = os.path.join(subdir, 'latents_and_labels.npz')
        if os.path.exists(npz_file):
            process_npz(npz_file, subdir, nu=args.nu)
        else:
            print(f"Skipping {entry}: no 'latents_and_labels.npz' found.")
