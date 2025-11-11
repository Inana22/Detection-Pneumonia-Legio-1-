"""
Pneumonia Detection System - Training Script
Classification Only (Pneumonia vs Normal)
Menggunakan DenseNet121 pretrained ImageNet
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============= KONFIGURASI =============
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
BASE_DIR = r'c:\paruparu\chest_xray\chest_xray'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Buat folder models jika belum ada
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("PNEUMONIA DETECTION SYSTEM - TRAINING")
print("=" * 60)
print(f"Base Directory: {BASE_DIR}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("=" * 60)

# ============= DATA AUGMENTATION =============
print("\nSetting up Data Augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,  # X-Ray tidak boleh di-flip horizontal
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# ============= LOAD DATA =============
print("\nLoading datasets...")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"\nClass indices: {train_generator.class_indices}")

# ============= BUILD MODEL =============
print("\nBuilding DenseNet121 Classification Model...")

def build_classification_model():
    """
    Build DenseNet121 model untuk klasifikasi Pneumonia
    """
    # Load DenseNet121 pretrained
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers (akan di-unfreeze nanti untuk fine-tuning)
    base_model.trainable = False
    
    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid', name='classification_output')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

model, base_model = build_classification_model()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

print("\nModel Summary:")
print(model.summary())
print(f"\nTotal parameters: {model.count_params():,}")

# ============= CALLBACKS =============
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, f'best_model_{timestamp}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============= TRAINING PHASE 1: Feature Extraction =============
print("\n" + "=" * 60)
print("PHASE 1: Training with frozen backbone")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Train awal dengan frozen backbone
    callbacks=callbacks,
    verbose=1
)

# ============= TRAINING PHASE 2: Fine-Tuning =============
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning with unfrozen layers")
print("=" * 60)

# Unfreeze beberapa layer terakhir dari base model
base_model.trainable = True

# Freeze layer awal, unfreeze layer akhir
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compile ulang dengan learning rate lebih kecil
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Continue training
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=20,
    callbacks=callbacks,
    verbose=1
)

# ============= SAVE FINAL MODEL =============
final_model_path = os.path.join(MODELS_DIR, f'final_model_{timestamp}.h5')
model.save(final_model_path)
print(f"\nFinal model saved: {final_model_path}")

# ============= EVALUATION =============
print("\n" + "=" * 60)
print("EVALUATING MODEL ON TEST SET")
print("=" * 60)

# Evaluate on test set
test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_generator, verbose=1)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test AUC-ROC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {2*(test_precision*test_recall)/(test_precision+test_recall):.4f}")

# Predictions
test_generator.reset()
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============= SAVE TRAINING HISTORY =============
# Combine both phases
history_combined = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'auc': history_phase1.history['auc'] + history_phase2.history['auc'],
    'val_auc': history_phase1.history['val_auc'] + history_phase2.history['val_auc'],
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall)
}

history_path = os.path.join(MODELS_DIR, f'training_history_{timestamp}.json')
with open(history_path, 'w') as f:
    json.dump(history_combined, f, indent=4)

print(f"\nTraining history saved: {history_path}")

# ============= PLOT TRAINING HISTORY =============
print("\nGenerating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training History - Pneumonia Detection Model', fontsize=16, fontweight='bold')

# Accuracy
axes[0, 0].plot(history_combined['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history_combined['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0, 0].set_title('Model Accuracy', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history_combined['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history_combined['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0, 1].set_title('Model Loss', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# AUC
axes[1, 0].plot(history_combined['auc'], label='Train AUC', linewidth=2)
axes[1, 0].plot(history_combined['val_auc'], label='Val AUC', linewidth=2)
axes[1, 0].axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1, 0].set_title('AUC-ROC Score', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'],
            cbar_kws={'label': 'Count'})
axes[1, 1].set_title('Confusion Matrix (Test Set)', fontweight='bold')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plot_path = os.path.join(MODELS_DIR, f'training_plot_{timestamp}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Training plots saved: {plot_path}")

# ============= SUMMARY =============
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nModel Location: {MODELS_DIR}")
print(f"Best Model: best_model_{timestamp}.h5")
print(f"Final Model: final_model_{timestamp}.h5")
print(f"Training History: training_history_{timestamp}.json")
print(f"Training Plot: training_plot_{timestamp}.png")
print("\nFinal Test Results:")
print(f"   Accuracy: {test_acc*100:.2f}%")
print(f"   AUC-ROC: {test_auc:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")
print("\nNext Step: Run 'python gui_app.py' to use the model!")
print("=" * 60)
