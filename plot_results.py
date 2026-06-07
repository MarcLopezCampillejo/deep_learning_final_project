"""
Script para graficar:
- Loss curves (entrenamiento vs test)
- Accuracy curves (entrenamiento vs test)
- Confusion matrix

Guarda todo en una carpeta de resultados.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuración
METRICS_FILE = 'checkpoints/training_metrics.csv'
CONFUSION_MATRIX_FILE = 'checkpoints/confusion_matrix.csv'
OUTPUT_DIR = 'training_results_vgg19'

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Carpeta creada: {OUTPUT_DIR}")

# Clases de emociones
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def split_epoch_runs(df):
    """Split appended metric logs so separate runs are never connected."""
    run_ids = df['epoch'].diff().lt(0).cumsum()
    return [run.copy() for _, run in df.groupby(run_ids, sort=False)]


def plot_metric_runs(ax, runs, y_column, label, marker, color):
    for run_index, run in enumerate(runs):
        run = run.sort_values('epoch', kind='mergesort')
        ax.plot(
            run['epoch'],
            run[y_column],
            marker=marker,
            linestyle='-',
            label=label if run_index == 0 else None,
            linewidth=2,
            markersize=4,
            color=color,
        )


def plot_existing_metrics(ax, runs, metric_specs):
    for y_column, label, marker, color in metric_specs:
        if y_column in runs[0].columns:
            plot_metric_runs(ax, runs, y_column, label, marker, color)


def label_last_metric_point(ax, runs, y_column, text, color, x_offset=0.25):
    if y_column not in runs[0].columns:
        return

    last_run = runs[-1].sort_values('epoch', kind='mergesort')
    last_point = last_run.dropna(subset=['epoch', y_column]).iloc[-1]
    ax.text(
        last_point['epoch'] + x_offset,
        last_point[y_column],
        text,
        color=color,
        fontsize=10,
        fontweight='bold',
        va='center',
        bbox={'facecolor': 'white', 'edgecolor': color, 'boxstyle': 'round,pad=0.25', 'alpha': 0.9},
        clip_on=False,
    )


# =========================================================================
# 1. Leer métricas de entrenamiento
# =========================================================================
print(f"\n📊 Leyendo {METRICS_FILE}...")
if not os.path.exists(METRICS_FILE):
    raise FileNotFoundError(
        f"No existe {METRICS_FILE}. Reentrena con mainpro_FER.py para generar este CSV."
    )
df_metrics = pd.read_csv(METRICS_FILE)
df_metrics['epoch'] = pd.to_numeric(df_metrics['epoch'], errors='coerce')
df_metrics = df_metrics.dropna(subset=['epoch']).reset_index(drop=True)
runs = split_epoch_runs(df_metrics)
if len(runs) > 1:
    print(f"Detectados {len(runs)} bloques de entrenamiento; se grafican sin unirlos entre si")
print(f"✓ Datos cargados: {len(df_metrics)} épocas")

# =========================================================================
# 2. Gráfico de LOSS
# =========================================================================
print("\n📈 Creando gráfico de Loss...")
fig, ax = plt.subplots(figsize=(12, 6))

plot_existing_metrics(ax, runs, [
    ('train_loss', 'Training Loss', 'o', 'C0'),
    ('test_loss', 'Test Loss', 's', 'C1'),
    ('public_test_loss', 'Public Test Loss (FER PublicTest)', 's', 'C1'),
    ('private_test_loss', 'Private Test Loss (FER PrivateTest)', '^', 'C2'),
])
label_last_metric_point(ax, runs, 'public_test_loss', 'Public Test', 'C1')
label_last_metric_point(ax, runs, 'private_test_loss', 'Private Test', 'C2')
if not df_metrics.empty:
    ax.set_xlim(df_metrics['epoch'].min(), df_metrics['epoch'].max() + 2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Loss Curves (FER2013 VGG19)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(OUTPUT_DIR, 'loss_curve.png')
plt.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {loss_path}")

# =========================================================================
# 3. Gráfico de ACCURACY
# =========================================================================
print("📈 Creando gráfico de Accuracy...")
fig, ax = plt.subplots(figsize=(12, 6))

plot_existing_metrics(ax, runs, [
    ('train_accuracy', 'Training Accuracy', 'o', 'C0'),
    ('train_acc', 'Training Accuracy', 'o', 'C0'),
    ('test_accuracy', 'Test Accuracy', 's', 'C1'),
    ('public_test_acc', 'Public Test Accuracy', 's', 'C1'),
    ('private_test_acc', 'Private Test Accuracy', '^', 'C2'),
    ('test_macro_accuracy', 'Test Macro Accuracy', 'D', 'C3'),
])

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Training vs Test Accuracy (VGG19)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
accuracy_path = os.path.join(OUTPUT_DIR, 'accuracy_curve.png')
plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {accuracy_path}")

# =========================================================================
# 4. Gráfico de LEARNING RATE (si varía)
# =========================================================================
if 'learning_rate' in df_metrics.columns and len(df_metrics['learning_rate'].unique()) > 1:
    print("📈 Creando gráfico de Learning Rate...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_metric_runs(ax, runs, 'learning_rate', 'Learning Rate', 'o', 'purple')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule (VGG19)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    lr_path = os.path.join(OUTPUT_DIR, 'learning_rate.png')
    plt.savefig(lr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {lr_path}")

# =========================================================================
# 5. Matriz de CONFUSIÓN
# =========================================================================
if os.path.exists(CONFUSION_MATRIX_FILE):
    print(f"\n📊 Leyendo {CONFUSION_MATRIX_FILE}...")
    cm = pd.read_csv(CONFUSION_MATRIX_FILE, index_col=0)
    
    print("📈 Creando Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set (VGG19)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {cm_path}")
    
    # Normalizada (porcentajes)
    cm_normalized = cm.div(cm.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'},
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized %) - Test Set (VGG19)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_norm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: {cm_norm_path}")
else:
    print(f"⚠️  No encontrado: {CONFUSION_MATRIX_FILE}")

# =========================================================================
# 6. Resumen de métricas
# =========================================================================
print("\n" + "="*60)
print("📊 RESUMEN DE MÉTRICAS")
print("="*60)

final_epoch = df_metrics.iloc[-1]
summary_columns = [
    ('train_loss', 'Train Loss', ''),
    ('train_accuracy', 'Train Acc', '%'),
    ('train_acc', 'Train Acc', '%'),
    ('test_loss', 'Test Loss', ''),
    ('test_accuracy', 'Test Acc', '%'),
    ('public_test_loss', 'Public Test Loss', ''),
    ('public_test_acc', 'Public Test Acc', '%'),
    ('private_test_loss', 'Private Test Loss', ''),
    ('private_test_acc', 'Private Test Acc', '%'),
    ('test_macro_accuracy', 'Test Macro Acc', '%'),
]
accuracy_priority = ['private_test_acc', 'public_test_acc', 'test_accuracy', 'test_macro_accuracy']
best_accuracy_column = next((column for column in accuracy_priority if column in df_metrics.columns), None)
best_accuracy_row = df_metrics.iloc[df_metrics[best_accuracy_column].idxmax()] if best_accuracy_column else None

print(f"\nUltima epoca ({int(final_epoch['epoch'])}):")
for column, label, suffix in summary_columns:
    if column in df_metrics.columns:
        print(f"  {label}: {final_epoch[column]:.4f}{suffix}")

if best_accuracy_column:
    print(f"\nMejor {best_accuracy_column} (epoca {int(best_accuracy_row['epoch'])}):")
    print(f"  {best_accuracy_column}: {best_accuracy_row[best_accuracy_column]:.2f}%")

summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("RESUMEN DE ENTRENAMIENTO - VGG19\n")
    f.write("="*60 + "\n\n")
    f.write(f"Archivo de metricas: {METRICS_FILE}\n\n")
    f.write(f"Ultima epoca ({int(final_epoch['epoch'])}):\n")
    for column, label, suffix in summary_columns:
        if column in df_metrics.columns:
            f.write(f"  {label}: {final_epoch[column]:.4f}{suffix}\n")
    if best_accuracy_column:
        f.write(f"\nMejor {best_accuracy_column} (epoca {int(best_accuracy_row['epoch'])}):\n")
        f.write(f"  {best_accuracy_column}: {best_accuracy_row[best_accuracy_column]:.2f}%\n")

print(f"Guardado: {summary_path}")
print("\n" + "="*60)
print(f"COMPLETADO. Todos los graficos guardados en: {OUTPUT_DIR}/")
print("="*60)
sys.exit(0)
best_test_acc_idx = df_metrics['test_accuracy'].idxmax()
best_test_acc_row = df_metrics.iloc[best_test_acc_idx]

print(f"\nÚltima época ({int(final_epoch['epoch'])}):")
print(f"  Train Loss:  {final_epoch['train_loss']:.4f}")
print(f"  Train Acc:   {final_epoch['train_accuracy']:.2f}%")
print(f"  Test Loss:   {final_epoch['test_loss']:.4f}")
print(f"  Test Acc:    {final_epoch['test_accuracy']:.2f}%")
print(f"  Test Macro Acc: {final_epoch['test_macro_accuracy']:.2f}%")

print(f"\nMejor Test Accuracy (época {int(best_test_acc_row['epoch'])}):")
print(f"  Test Acc:    {best_test_acc_row['test_accuracy']:.2f}%")
print(f"  Test Macro Acc: {best_test_acc_row['test_macro_accuracy']:.2f}%")

print(f"\nMejor Test Macro Accuracy:")
best_macro_idx = df_metrics['test_macro_accuracy'].idxmax()
best_macro_row = df_metrics.iloc[best_macro_idx]
print(f"  Época:       {int(best_macro_row['epoch'])}")
print(f"  Test Acc:    {best_macro_row['test_accuracy']:.2f}%")
print(f"  Test Macro Acc: {best_macro_row['test_macro_accuracy']:.2f}%")

# Guardar resumen
summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("RESUMEN DE ENTRENAMIENTO - VGG19\n")
    f.write("="*60 + "\n\n")
    f.write(f"Última época ({int(final_epoch['epoch'])}):\n")
    f.write(f"  Train Loss:  {final_epoch['train_loss']:.4f}\n")
    f.write(f"  Train Acc:   {final_epoch['train_accuracy']:.2f}%\n")
    f.write(f"  Test Loss:   {final_epoch['test_loss']:.4f}\n")
    f.write(f"  Test Acc:    {final_epoch['test_accuracy']:.2f}%\n")
    f.write(f"  Test Macro Acc: {final_epoch['test_macro_accuracy']:.2f}%\n\n")
    f.write(f"Mejor Test Accuracy (época {int(best_test_acc_row['epoch'])}):\n")
    f.write(f"  Test Acc:    {best_test_acc_row['test_accuracy']:.2f}%\n")
    f.write(f"  Test Macro Acc: {best_test_acc_row['test_macro_accuracy']:.2f}%\n\n")
    f.write(f"Mejor Test Macro Accuracy (época {int(best_macro_row['epoch'])}):\n")
    f.write(f"  Test Acc:    {best_macro_row['test_accuracy']:.2f}%\n")
    f.write(f"  Test Macro Acc: {best_macro_row['test_macro_accuracy']:.2f}%\n")

print(f"✓ Guardado: {summary_path}")

print("\n" + "="*60)
print(f"✓ ¡COMPLETADO! Todos los gráficos guardados en: {OUTPUT_DIR}/")
print("="*60)
