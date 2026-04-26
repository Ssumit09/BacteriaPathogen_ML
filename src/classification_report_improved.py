"""
classification_report_by_agar.py

Enhanced classification report with stratified analysis by agar type.
Shows overall metrics + per-agar breakdown to identify which agar medium
performs best for B. pseudomallei detection.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from data_loader_fixed import get_datasets
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = Path("models/final_finetuned_model.keras")
VAL_CSV = Path("metadata/splits_fixed/val.csv")
REPORT_SAVE_PATH = Path("outputs/classification_reports_by_agar/")
REPORT_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ===========================
# LOAD MODEL & DATASETS
# ===========================
print("📂 Loading model and datasets...")
model = load_model(str(MODEL_PATH), compile=False)
train_ds, val_ds = get_datasets()

# Load validation metadata
val_metadata = pd.read_csv(VAL_CSV)

# ===========================
# GENERATE PREDICTIONS
# ===========================
def get_predictions_and_labels(dataset, dataset_name=""):
    """Extract predictions and true labels from dataset"""
    print(f"\n🔄 Generating predictions for {dataset_name}...")
    
    all_predictions = []
    all_labels = []
    
    for (images, metadata), labels in dataset:
        predictions = model.predict([images, metadata], verbose=0)
        predictions_binary = (predictions > 0.5).astype(int).flatten()
        
        all_predictions.extend(predictions_binary)
        all_labels.extend(labels.numpy().astype(int))
    
    return np.array(all_predictions, dtype=int), np.array(all_labels, dtype=int)

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)
val_pred, val_labels = get_predictions_and_labels(val_ds, "Validation Set")

# ===========================
# MERGE PREDICTIONS WITH METADATA
# ===========================
print("\n📊 Merging predictions with metadata...")
val_metadata['predicted_label'] = val_pred
val_metadata['true_label'] = val_labels
val_metadata['correct_prediction'] = (val_pred == val_labels)

# ===========================
# OVERALL METRICS
# ===========================
def print_overall_metrics(y_true, y_pred, dataset_name):
    """Print overall classification metrics"""
    print("\n" + "="*80)
    print(f"{dataset_name} - OVERALL METRICS")
    print("="*80)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "-"*80)
    print("Class-wise Breakdown:")
    print("-"*80)
    
    report = classification_report(
        y_true, y_pred, 
        target_names=['0 (Other Bacteria)', '1 (B. pseudomallei)'],
        digits=4
    )
    print(report)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

overall_metrics = print_overall_metrics(val_labels, val_pred, "VALIDATION SET")

# ===========================
# PER-AGAR METRICS
# ===========================
def calculate_per_agar_metrics(df):
    """Calculate metrics for each agar type"""
    print("\n" + "="*80)
    print("PER-AGAR TYPE METRICS")
    print("="*80)
    
    agar_types = df['agar'].unique()
    agar_metrics = []
    
    for agar in sorted(agar_types):
        agar_df = df[df['agar'] == agar]
        y_true = agar_df['true_label'].values
        y_pred = agar_df['predicted_label'].values
        
        n_samples = len(agar_df)
        n_bpseudo = (y_true == 1).sum()
        n_other = (y_true == 0).sum()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"\n{'='*60}")
        print(f"AGAR: {agar}")
        print(f"{'='*60}")
        print(f"Total samples: {n_samples}")
        print(f"  - B. pseudomallei: {n_bpseudo}")
        print(f"  - Other bacteria:  {n_other}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:3d}  |  FP: {cm[0,1]:3d}")
        print(f"  FN: {cm[1,0]:3d}  |  TP: {cm[1,1]:3d}")
        
        agar_metrics.append({
            'agar': agar,
            'n_samples': n_samples,
            'n_bpseudo': n_bpseudo,
            'n_other': n_other,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'TP': cm[1,1] if cm.shape == (2,2) else 0,
            'TN': cm[0,0] if cm.shape == (2,2) else 0,
            'FP': cm[0,1] if cm.shape == (2,2) else 0,
            'FN': cm[1,0] if cm.shape == (2,2) else 0
        })
    
    return pd.DataFrame(agar_metrics)

agar_metrics_df = calculate_per_agar_metrics(val_metadata)

# ===========================
# SAVE METRICS TO CSV
# ===========================
print("\n" + "="*80)
print("SAVING REPORTS")
print("="*80)

# Save per-agar metrics
agar_csv_path = REPORT_SAVE_PATH / "per_agar_metrics.csv"
agar_metrics_df.to_csv(agar_csv_path, index=False)
print(f"✅ Per-agar metrics saved to: {agar_csv_path}")

# Save detailed predictions with metadata
detailed_csv_path = REPORT_SAVE_PATH / "detailed_predictions.csv"
val_metadata.to_csv(detailed_csv_path, index=False)
print(f"✅ Detailed predictions saved to: {detailed_csv_path}")

# ===========================
# VISUALIZATIONS
# ===========================
print("\n📊 Generating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Accuracy Comparison
ax1 = fig.add_subplot(gs[0, 0])
agar_order = agar_metrics_df.sort_values('accuracy', ascending=False)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(agar_order)))
bars = ax1.bar(agar_order['agar'], agar_order['accuracy'], color=colors, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_xlabel('Agar Type', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy by Agar Type', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.axhline(y=overall_metrics['accuracy'], color='red', linestyle='--', 
            linewidth=2, label=f"Overall: {overall_metrics['accuracy']:.3f}")
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Precision Comparison
ax2 = fig.add_subplot(gs[0, 1])
agar_order = agar_metrics_df.sort_values('precision', ascending=False)
bars = ax2.bar(agar_order['agar'], agar_order['precision'], color=colors, edgecolor='black')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_xlabel('Agar Type', fontsize=12, fontweight='bold')
ax2.set_title('Precision by Agar Type', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1.0])
ax2.axhline(y=overall_metrics['precision'], color='red', linestyle='--', 
            linewidth=2, label=f"Overall: {overall_metrics['precision']:.3f}")
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Recall Comparison
ax3 = fig.add_subplot(gs[1, 0])
agar_order = agar_metrics_df.sort_values('recall', ascending=False)
bars = ax3.bar(agar_order['agar'], agar_order['recall'], color=colors, edgecolor='black')
ax3.set_ylabel('Recall', fontsize=12, fontweight='bold')
ax3.set_xlabel('Agar Type', fontsize=12, fontweight='bold')
ax3.set_title('Recall by Agar Type', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 1.0])
ax3.axhline(y=overall_metrics['recall'], color='red', linestyle='--', 
            linewidth=2, label=f"Overall: {overall_metrics['recall']:.3f}")
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. F1-Score Comparison
ax4 = fig.add_subplot(gs[1, 1])
agar_order = agar_metrics_df.sort_values('f1_score', ascending=False)
bars = ax4.bar(agar_order['agar'], agar_order['f1_score'], color=colors, edgecolor='black')
ax4.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax4.set_xlabel('Agar Type', fontsize=12, fontweight='bold')
ax4.set_title('F1-Score by Agar Type', fontsize=14, fontweight='bold')
ax4.set_ylim([0, 1.0])
ax4.axhline(y=overall_metrics['f1_score'], color='red', linestyle='--', 
            linewidth=2, label=f"Overall: {overall_metrics['f1_score']:.3f}")
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Sample Distribution
ax5 = fig.add_subplot(gs[2, 0])
x = np.arange(len(agar_metrics_df))
width = 0.35
bars1 = ax5.bar(x - width/2, agar_metrics_df['n_bpseudo'], width, 
                label='B. pseudomallei', color='coral', edgecolor='black')
bars2 = ax5.bar(x + width/2, agar_metrics_df['n_other'], width,
                label='Other Bacteria', color='skyblue', edgecolor='black')
ax5.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax5.set_xlabel('Agar Type', fontsize=12, fontweight='bold')
ax5.set_title('Sample Distribution by Agar Type', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(agar_metrics_df['agar'])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 6. Grouped Bar Chart - All Metrics
ax6 = fig.add_subplot(gs[2, 1])
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(agar_metrics_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    offset = width * (i - 1.5)
    bars = ax6.bar(x + offset, agar_metrics_df[metric], width, 
                   label=metric.replace('_', ' ').title())

ax6.set_ylabel('Score', fontsize=12, fontweight='bold')
ax6.set_xlabel('Agar Type', fontsize=12, fontweight='bold')
ax6.set_title('All Metrics Comparison by Agar Type', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(agar_metrics_df['agar'])
ax6.legend(loc='lower right')
ax6.set_ylim([0, 1.0])
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('B. pseudomallei Classification Performance by Agar Type', 
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
viz_path = REPORT_SAVE_PATH / "per_agar_metrics_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {viz_path}")
plt.close()

# ===========================
# CONFUSION MATRICES PER AGAR
# ===========================
print("\n📊 Generating confusion matrices...")

agar_types = sorted(val_metadata['agar'].unique())
n_agars = len(agar_types)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, agar in enumerate(agar_types):
    agar_df = val_metadata[val_metadata['agar'] == agar]
    y_true = agar_df['true_label'].values
    y_pred = agar_df['predicted_label'].values
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy for this agar
    agar_acc = accuracy_score(y_true, y_pred)
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Other', 'B. pseudomallei'],
                yticklabels=['Other', 'B. pseudomallei'],
                ax=axes[idx], cbar=True, square=True,
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    axes[idx].set_title(f'{agar} Agar\nAccuracy: {agar_acc:.3f}', 
                        fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrices by Agar Type', fontsize=16, fontweight='bold')
plt.tight_layout()

cm_path = REPORT_SAVE_PATH / "confusion_matrices_by_agar.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrices saved to: {cm_path}")
plt.close()

# ===========================
# RANKING TABLE
# ===========================
print("\n" + "="*80)
print("AGAR TYPE RANKING")
print("="*80)

ranking_df = agar_metrics_df.copy()
ranking_df['rank_accuracy'] = ranking_df['accuracy'].rank(ascending=False)
ranking_df['rank_precision'] = ranking_df['precision'].rank(ascending=False)
ranking_df['rank_recall'] = ranking_df['recall'].rank(ascending=False)
ranking_df['rank_f1'] = ranking_df['f1_score'].rank(ascending=False)
ranking_df['avg_rank'] = ranking_df[['rank_accuracy', 'rank_precision', 
                                      'rank_recall', 'rank_f1']].mean(axis=1)

ranking_df = ranking_df.sort_values('avg_rank')

print("\nRanking by Average Performance:")
print("-" * 80)
for idx, row in ranking_df.iterrows():
    print(f"{int(row['avg_rank'])}. {row['agar']:20s} | "
          f"Acc: {row['accuracy']:.3f} | Prec: {row['precision']:.3f} | "
          f"Rec: {row['recall']:.3f} | F1: {row['f1_score']:.3f}")

# Save ranking
ranking_path = REPORT_SAVE_PATH / "agar_ranking.csv"
ranking_df.to_csv(ranking_path, index=False)
print(f"\n✅ Ranking saved to: {ranking_path}")

# ===========================
# SUMMARY REPORT
# ===========================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

best_agar = ranking_df.iloc[0]
worst_agar = ranking_df.iloc[-1]

print(f"\n🏆 BEST PERFORMING AGAR: {best_agar['agar']}")
print(f"   - Accuracy:  {best_agar['accuracy']:.3f}")
print(f"   - Precision: {best_agar['precision']:.3f}")
print(f"   - Recall:    {best_agar['recall']:.3f}")
print(f"   - F1-Score:  {best_agar['f1_score']:.3f}")

print(f"\n⚠️  WORST PERFORMING AGAR: {worst_agar['agar']}")
print(f"   - Accuracy:  {worst_agar['accuracy']:.3f}")
print(f"   - Precision: {worst_agar['precision']:.3f}")
print(f"   - Recall:    {worst_agar['recall']:.3f}")
print(f"   - F1-Score:  {worst_agar['f1_score']:.3f}")

print("\n💡 RECOMMENDATIONS:")
print("-" * 80)

# Check for problematic agars (accuracy < 0.8)
problematic = agar_metrics_df[agar_metrics_df['accuracy'] < 0.8]
if len(problematic) > 0:
    print("⚠️  The following agar types show lower performance (<80% accuracy):")
    for _, row in problematic.iterrows():
        print(f"   - {row['agar']}: {row['accuracy']:.3f} accuracy")
    print("\n   Consider:")
    print("   1. Collecting more training samples for these agar types")
    print("   2. Reviewing image quality for these specific agars")
    print("   3. Investigating if these agars are inherently harder to classify")
else:
    print("✅ All agar types show good performance (>80% accuracy)")

# Check for imbalanced samples
print("\n📊 Sample Distribution Analysis:")
for _, row in agar_metrics_df.iterrows():
    imbalance_ratio = row['n_bpseudo'] / row['n_other'] if row['n_other'] > 0 else float('inf')
    if imbalance_ratio < 0.5 or imbalance_ratio > 2.0:
        print(f"   ⚠️  {row['agar']}: Imbalanced samples "
              f"(B.pseudo: {row['n_bpseudo']}, Other: {row['n_other']})")

print("\n" + "="*80)
print(f"📁 All reports saved to: {REPORT_SAVE_PATH}")
print("="*80)
print("\n✅ Analysis complete!")