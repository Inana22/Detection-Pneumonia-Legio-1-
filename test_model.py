"""
Pneumonia Detection System - Testing Script
Evaluasi model yang sudah di-training pada test dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# ============= KONFIGURASI =============
IMG_SIZE = 224
BATCH_SIZE = 16
BASE_DIR = r'c:\paruparu\chest_xray\chest_xray'
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Buat folder results jika belum ada
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("PNEUMONIA DETECTION SYSTEM - MODEL TESTING")
print("=" * 60)

# ============= LOAD MODEL =============
print("\nMencari model...")

# Cari model terbaru
model_files = glob.glob(os.path.join(MODELS_DIR, 'best_model_*.h5'))
if not model_files:
    model_files = glob.glob(os.path.join(MODELS_DIR, 'final_model_*.h5'))

if not model_files:
    print("ERROR: Tidak ada model ditemukan!")
    print(f"   Pastikan sudah menjalankan train_model.py terlebih dahulu")
    print(f"   Model harus berada di: {MODELS_DIR}")
    exit()

# Gunakan model terbaru
model_path = max(model_files, key=os.path.getctime)
print(f"Model ditemukan: {os.path.basename(model_path)}")

print("\nLoading model...")
model = load_model(model_path)
print("Model berhasil di-load!")

# ============= LOAD TEST DATA =============
print("\nLoading test dataset...")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Test samples: {test_generator.samples}")
print(f"Class indices: {test_generator.class_indices}")

# ============= EVALUATION =============
print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

# Evaluate
results = model.evaluate(test_generator, verbose=1)
metric_names = model.metrics_names

print("\nTest Results:")
for name, value in zip(metric_names, results):
    if 'loss' in name:
        print(f"   {name.capitalize()}: {value:.4f}")
    else:
        print(f"   {name.capitalize()}: {value*100:.2f}%" if value <= 1 else f"   {name.capitalize()}: {value:.4f}")

# ============= PREDICTIONS =============
print("\nMaking predictions...")

test_generator.reset()
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = test_generator.classes

# ============= DETAILED METRICS =============
print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION METRICS")
print("=" * 60)

# Classification Report
print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'], digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(f"                  Predicted")
print(f"              NORMAL  PNEUMONIA")
print(f"Actual NORMAL    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       PNEUMONIA {cm[1][0]:4d}    {cm[1][1]:4d}")

# Calculate detailed metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# AUC-ROC
auc_score = roc_auc_score(y_true, y_pred_proba)

print("\nAdditional Metrics:")
print(f"   True Positives (TP): {tp}")
print(f"   True Negatives (TN): {tn}")
print(f"   False Positives (FP): {fp}")
print(f"   False Negatives (FN): {fn}")
print(f"\n   Accuracy: {accuracy*100:.2f}%")
print(f"   Precision: {precision:.4f}")
print(f"   Recall (Sensitivity): {recall:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   F1-Score: {f1_score:.4f}")
print(f"   AUC-ROC: {auc_score:.4f}")

# ============= VISUALIZATION =============
print("\nGenerating visualizations...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Pneumonia Detection Model - Test Results', fontsize=20, fontweight='bold', y=0.98)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'],
            cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
ax1.set_ylabel('True Label', fontweight='bold')
ax1.set_xlabel('Predicted Label', fontweight='bold')

# 2. Normalized Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'],
            cbar_kws={'label': 'Percentage'})
ax2.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=14)
ax2.set_ylabel('True Label', fontweight='bold')
ax2.set_xlabel('Predicted Label', fontweight='bold')

# 3. ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
ax3.plot(fpr, tpr, linewidth=3, label=f'AUC = {auc_score:.4f}', color='#2E86AB')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC = 0.5)')
ax3.set_xlabel('False Positive Rate', fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontweight='bold')
ax3.set_title('ROC Curve', fontweight='bold', fontsize=14)
ax3.legend(loc='lower right', fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Metrics Bar Chart
ax4 = fig.add_subplot(gs[1, 0])
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
values = [accuracy, precision, recall, specificity, f1_score]
colors = ['#06A77D', '#1192E8', '#FA4D56', '#F1C21B', '#A56EFF']
bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Score', fontweight='bold')
ax4.set_title('Performance Metrics', fontweight='bold', fontsize=14)
ax4.set_ylim([0, 1])
ax4.grid(True, axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold')
ax4.tick_params(axis='x', rotation=45)

# 5. Prediction Distribution
ax5 = fig.add_subplot(gs[1, 1])
# Histogram untuk NORMAL dan PNEUMONIA
normal_probs = y_pred_proba[y_true == 0]
pneumonia_probs = y_pred_proba[y_true == 1]
ax5.hist(normal_probs, bins=30, alpha=0.6, label='Normal', color='#06A77D', edgecolor='black')
ax5.hist(pneumonia_probs, bins=30, alpha=0.6, label='Pneumonia', color='#FA4D56', edgecolor='black')
ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax5.set_xlabel('Predicted Probability', fontweight='bold')
ax5.set_ylabel('Frequency', fontweight='bold')
ax5.set_title('Prediction Distribution', fontweight='bold', fontsize=14)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Class Distribution (Test Set)
ax6 = fig.add_subplot(gs[1, 2])
class_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
wedges, texts, autotexts = ax6.pie(class_counts, labels=['NORMAL', 'PNEUMONIA'],
                                     autopct='%1.1f%%', startangle=90,
                                     colors=['#06A77D', '#FA4D56'],
                                     textprops={'fontweight': 'bold', 'fontsize': 12})
ax6.set_title('Test Set Class Distribution', fontweight='bold', fontsize=14)

# 7. Sample Predictions (Correct)
ax7 = fig.add_subplot(gs[2, :2])
ax7.axis('off')
ax7.text(0.5, 0.95, 'Sample Predictions', ha='center', va='top', 
         fontweight='bold', fontsize=14, transform=ax7.transAxes)

# Get some sample images
test_generator.reset()
sample_images, sample_labels = next(test_generator)
sample_preds = model.predict(sample_images[:6])

# Create mini subplot for sample images
for i in range(6):
    ax_sample = fig.add_subplot(gs[2, i//3], frameon=False)
    if i >= 3:
        ax_sample = fig.add_subplot(gs[2, 1], frameon=False)
    
# Actually show 6 samples in a row
gs_samples = gs[2, :].subgridspec(1, 6, wspace=0.3)
for i in range(6):
    ax_sample = fig.add_subplot(gs_samples[i])
    ax_sample.imshow(sample_images[i])
    pred_class = 'PNEUMONIA' if sample_preds[i] > 0.5 else 'NORMAL'
    true_class = 'PNEUMONIA' if sample_labels[i] > 0.5 else 'NORMAL'
    color = 'green' if pred_class == true_class else 'red'
    ax_sample.set_title(f'True: {true_class}\nPred: {pred_class}\n({sample_preds[i][0]*100:.1f}%)',
                       fontsize=8, color=color, fontweight='bold')
    ax_sample.axis('off')

# 8. Model Info Text
info_text = f"""
MODEL INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: {os.path.basename(model_path)}
Test Samples: {test_generator.samples}
Image Size: {IMG_SIZE}x{IMG_SIZE}
Batch Size: {BATCH_SIZE}

PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy: {accuracy*100:.2f}%
AUC-ROC: {auc_score:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1_score:.4f}

CONFUSION MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
True Positives: {tp}
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
"""

# Save plot
plot_path = os.path.join(RESULTS_DIR, f'test_results_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_path}")

# ============= SAVE RESULTS TO TEXT FILE =============
results_text_path = os.path.join(RESULTS_DIR, f'test_results_{timestamp}.txt')
with open(results_text_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("PNEUMONIA DETECTION SYSTEM - TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {os.path.basename(model_path)}\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Test Samples: {test_generator.samples}\n\n")
    f.write("=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("=" * 60 + "\n")
    f.write(f"                  Predicted\n")
    f.write(f"              NORMAL  PNEUMONIA\n")
    f.write(f"Actual NORMAL    {cm[0][0]:4d}    {cm[0][1]:4d}\n")
    f.write(f"       PNEUMONIA {cm[1][0]:4d}    {cm[1][1]:4d}\n\n")
    f.write("=" * 60 + "\n")
    f.write("DETAILED METRICS\n")
    f.write("=" * 60 + "\n")
    f.write(f"True Positives (TP): {tp}\n")
    f.write(f"True Negatives (TN): {tn}\n")
    f.write(f"False Positives (FP): {fp}\n")
    f.write(f"False Negatives (FN): {fn}\n\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"F1-Score: {f1_score:.4f}\n")
    f.write(f"AUC-ROC: {auc_score:.4f}\n")

print(f"Results text saved: {results_text_path}")

# Show plot
plt.show()

# ============= SUMMARY =============
print("\n" + "=" * 60)
print("TESTING COMPLETED!")
print("=" * 60)
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Visualization: test_results_{timestamp}.png")
print(f"Text Report: test_results_{timestamp}.txt")
print("\nQuick Summary:")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   AUC-ROC: {auc_score:.4f}")
print(f"   F1-Score: {f1_score:.4f}")
print("=" * 60)
