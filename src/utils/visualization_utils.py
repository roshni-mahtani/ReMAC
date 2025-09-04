import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc
)

def plot_training_results(
    fold,
    train_total_loss,
    val_total_loss,
    train_recon_loss,
    val_recon_loss,
    train_class_loss,
    val_class_loss,
    y_true,
    y_pred,
    output_path,
    class_labels=['Nevus', 'Melanoma']
):
    """
    Genera y guarda las gráficas de loss, matriz de confusión, curva ROC y curva PR para un fold.
    """
    fig, axs = plt.subplots(3, 2, figsize=(18, 15))  # Ahora 3 filas x 2 columnas

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, (np.array(y_pred) >= 0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=axs[0, 0], values_format='d')
    axs[0, 0].set_title(f'Fold {fold+1} Validation Confusion Matrix')

    # --- Total Loss ---
    axs[0, 1].plot(range(1, len(train_total_loss)+1), train_total_loss, label='Train Loss')
    axs[0, 1].plot(range(1, len(val_total_loss)+1), val_total_loss, label='Val Loss')
    axs[0, 1].set_title("Total Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- Reconstruction Loss ---
    axs[1, 0].plot(range(1, len(train_recon_loss)+1), train_recon_loss, label='Train Recon Loss')
    axs[1, 0].plot(range(1, len(val_recon_loss)+1), val_recon_loss, label='Val Recon Loss')
    axs[1, 0].set_title("Reconstruction Loss")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Classification Loss ---
    axs[1, 1].plot(range(1, len(train_class_loss)+1), train_class_loss, label='Train Class Loss')
    axs[1, 1].plot(range(1, len(val_class_loss)+1), val_class_loss, label='Val Class Loss')
    axs[1, 1].set_title("Classification Loss")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    axs[2, 0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axs[2, 0].plot([0, 1], [0, 1], 'k--')
    axs[2, 0].set_xlim([0.0, 1.0])
    axs[2, 0].set_ylim([0.0, 1.05])
    axs[2, 0].set_xlabel('False Positive Rate')
    axs[2, 0].set_ylabel('True Positive Rate')
    axs[2, 0].set_title('ROC Curve')
    axs[2, 0].legend(loc="lower right")
    axs[2, 0].grid(True)

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    axs[2, 1].plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    axs[2, 1].set_xlim([0.0, 1.0])
    axs[2, 1].set_ylim([0.0, 1.05])
    axs[2, 1].set_xlabel('Recall')
    axs[2, 1].set_ylabel('Precision')
    axs[2, 1].set_title('Precision-Recall Curve')
    axs[2, 1].legend(loc="lower left")
    axs[2, 1].grid(True)

    # Guardar figura
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# PARA ML
def plot_ml_results(
    fold,
    y_true,
    y_pred_prob,
    output_path,
    class_labels=['Nevus', 'Melanoma']
):
    """
    Guarda matriz de confusión, curva ROC y curva PR para un fold de un modelo ML clásico.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # 1 fila, 3 columnas

    # Confusion Matrix
    cm = confusion_matrix(y_true, (np.array(y_pred_prob) >= 0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=axs[0], values_format='d')
    axs[0].set_title(f'Fold {fold+1} - Confusion Matrix')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlabel('FPR')
    axs[1].set_ylabel('TPR')
    axs[1].set_title('ROC Curve')
    axs[1].legend(loc='lower right')

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    axs[2].plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    axs[2].set_xlabel('Recall')
    axs[2].set_ylabel('Precision')
    axs[2].set_title('Precision-Recall Curve')
    axs[2].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
