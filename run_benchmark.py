import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.radiomics.extractor import RadiomicExtractor
from src.deep_features.backbones import UltrasoundDeepFeatureExtractor
from src.fusion.models import (
    GatedAttentionFusionModel, EarlyFusionModel,
    DeepOnlyModel, RadiomicsOnlyModel, LateFusionEnsemble
)
from src.fusion.ordinal_loss import OrdinalHeckmattLoss
from src.segmentation.models import AttentionUNet, UNet
from src.segmentation.losses import DiceBCELoss
from src.segmentation.metrics import compute_segmentation_metrics
from src.evaluation.metrics import compute_classification_metrics

def main():
    print("=" * 70)
    print("HYBRID DEEP LEARNING & RADIOMIC BENCHMARK FOR FSHD MUSCLE ASSESSMENT")
    print("=" * 70)

    # 1. Load clinical dataset
    df_linked = pd.read_csv('data/fshd/linked_dataset.csv')
    df_valid = df_linked.dropna(subset=['heckmatt_grade']).copy()
    df_valid['heckmatt_grade'] = df_valid['heckmatt_grade'].astype(int)
    # Map Heckmatt grades (1, 2, 3, 4) to 0-indexed classes (0, 1, 2, 3)
    df_valid['target'] = df_valid['heckmatt_grade'] - 1
    print(f"Total valid Heckmatt-annotated scans: {len(df_valid)}")
    print("Class distribution:")
    print(df_valid['heckmatt_grade'].value_counts().sort_index())

    # 2. Extract / Load Radiomic and Deep Features
    feat_cache_file = 'data/fshd/precomputed_radiomics.csv'
    deep_cache_file = 'data/fshd/precomputed_deep.npy'

    extractor = RadiomicExtractor()
    deep_model = UltrasoundDeepFeatureExtractor(latent_dim=128)
    deep_model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep_model.to(device)

    # Check if precomputed features exist
    if os.path.exists(feat_cache_file) and os.path.exists(deep_cache_file):
        print("\nLoading precomputed radiomic and deep features from cache...")
        df_radiomics = pd.read_csv(feat_cache_file)
        deep_features = np.load(deep_cache_file)
    else:
        print("\nExtracting radiomics and deep representations for all cohort scans...")
        rad_rows = []
        deep_list = []
        import cv2

        t0 = time.time()
        for idx, row in df_valid.iterrows():
            img_path = os.path.join('data/fshd/images', row['filename'])
            mask_path = os.path.join('data/fshd/masks/masks', row['filename'])

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # Extract 48 standardized clinical radiomics
            feats = extractor.extract_features(
                img, mask,
                age=row['age'],
                sex=row['sex'],
                bmi=row['bmi'],
                muscle_code=row['muscle_code']
            )
            feats['filename'] = row['filename']
            feats['subject_id'] = row['subject_id']
            feats['heckmatt_grade'] = row['heckmatt_grade']
            feats['target'] = row['target']
            rad_rows.append(feats)

            # Extract deep representation from muscle ROI
            masked_img = (img * (mask > 0)).astype(np.float32) / 255.0
            # Resize to 256x256 for fast inference
            img_resized = cv2.resize(masked_img, (256, 256))
            tensor_in = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = deep_model(tensor_in).cpu().numpy()[0]
            deep_list.append(latent)

        df_radiomics = pd.DataFrame(rad_rows)
        deep_features = np.array(deep_list)

        df_radiomics.to_csv(feat_cache_file, index=False)
        np.save(deep_cache_file, deep_features)
        print(f"Extraction completed in {time.time()-t0:.2f}s! Saved features to cache.")

    # 3. Patient-Level Group K-Fold Cross Validation
    feature_cols = [c for c in df_radiomics.columns if c not in ['filename', 'subject_id', 'heckmatt_grade', 'target']]
    # Filter for the 48 baseline clinical features
    import joblib
    try:
        author_m = joblib.load('data/fshd/xgb_models/XGB_models/best_model_fold1.pkl')
        expected_48 = list(author_m.feature_names_in_)
    except Exception:
        expected_48 = feature_cols[:48]

    X_rad = df_radiomics[expected_48].values
    X_deep = deep_features
    y = df_radiomics['target'].values
    groups = df_radiomics['subject_id'].values

    # Handle NaNs if any
    X_rad = np.nan_to_num(X_rad, nan=0.0)

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\nRunning {n_splits}-Fold Patient-Level Stratified Cross Validation...")
    print(f"Feature dimensions: Radiomics={X_rad.shape[1]}, Deep Latent={X_deep.shape[1]}")

    results = {
        'Baseline_XGBoost_Radiomics': [],
        'Baseline_RandomForest_Radiomics': [],
        'Deep_ResNet_Only': [],
        'Early_Fusion': [],
        'Late_Fusion_Ensemble': [],
        'Proposed_Gated_Attention_Hybrid': []
    }

    fold = 1
    best_hybrid_weights = None
    best_qwk = -1.0

    for train_idx, val_idx in gkf.split(X_rad, y, groups=groups):
        print(f"\n--- Fold {fold}/{n_splits} (Train subjects: {len(np.unique(groups[train_idx]))}, Val subjects: {len(np.unique(groups[val_idx]))}) ---")
        
        # Scale radiomics
        scaler = StandardScaler()
        X_rad_train = scaler.fit_transform(X_rad[train_idx])
        X_rad_val = scaler.transform(X_rad[val_idx])

        X_deep_train = X_deep[train_idx]
        X_deep_val = X_deep[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Model 1: XGBoost on Radiomics (Marzola baseline)
        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective='multi:softprob', num_class=4, eval_metric='mlogloss',
            random_state=42
        )
        xgb.fit(X_rad_train, y_train)
        p_xgb = xgb.predict_proba(X_rad_val)
        y_pred_xgb = p_xgb.argmax(axis=1)
        results['Baseline_XGBoost_Radiomics'].append(compute_classification_metrics(y_val, y_pred_xgb, p_xgb))

        # Model 2: Random Forest on Radiomics
        rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)
        rf.fit(X_rad_train, y_train)
        p_rf = rf.predict_proba(X_rad_val)
        y_pred_rf = p_rf.argmax(axis=1)
        results['Baseline_RandomForest_Radiomics'].append(compute_classification_metrics(y_val, y_pred_rf, p_rf))

        # PyTorch Tensors
        t_deep_tr = torch.tensor(X_deep_train, dtype=torch.float32).to(device)
        t_deep_va = torch.tensor(X_deep_val, dtype=torch.float32).to(device)
        t_rad_tr = torch.tensor(X_rad_train, dtype=torch.float32).to(device)
        t_rad_va = torch.tensor(X_rad_val, dtype=torch.float32).to(device)
        t_y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
        t_y_va = torch.tensor(y_val, dtype=torch.long).to(device)

        # Class weights for loss
        class_counts = np.bincount(y_train, minlength=4)
        total_samples = len(y_train)
        weights = total_samples / (4.0 * np.maximum(class_counts, 1))
        class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

        # Model 3: Deep ResNet Only
        m_deep = DeepOnlyModel(deep_dim=128, num_classes=4).to(device)
        opt_deep = torch.optim.AdamW(m_deep.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=class_weights_t)
        m_deep.train()
        for epoch in range(60):
            opt_deep.zero_grad()
            out = m_deep(t_deep_tr)
            loss = crit(out['logits'], t_y_tr)
            loss.backward()
            opt_deep.step()
        m_deep.eval()
        with torch.no_grad():
            out_deep = m_deep(t_deep_va)
            p_deep = out_deep['probabilities'].cpu().numpy()
            y_pred_deep = p_deep.argmax(axis=1)
        results['Deep_ResNet_Only'].append(compute_classification_metrics(y_val, y_pred_deep, p_deep))

        # Model 4: Early Fusion
        m_early = EarlyFusionModel(deep_dim=128, radiomic_dim=len(expected_48), num_classes=4).to(device)
        opt_early = torch.optim.AdamW(m_early.parameters(), lr=1e-3, weight_decay=1e-4)
        m_early.train()
        for epoch in range(60):
            opt_early.zero_grad()
            out = m_early(t_deep_tr, t_rad_tr)
            loss = crit(out['logits'], t_y_tr)
            loss.backward()
            opt_early.step()
        m_early.eval()
        with torch.no_grad():
            out_early = m_early(t_deep_va, t_rad_va)
            p_early = out_early['probabilities'].cpu().numpy()
            y_pred_early = p_early.argmax(axis=1)
        results['Early_Fusion'].append(compute_classification_metrics(y_val, y_pred_early, p_early))

        # Model 5: Late Fusion Ensemble (Blend deep + radiomics)
        p_late = 0.5 * p_deep + 0.5 * p_xgb
        y_pred_late = p_late.argmax(axis=1)
        results['Late_Fusion_Ensemble'].append(compute_classification_metrics(y_val, y_pred_late, p_late))

        # Model 6: Proposed Gated Attention Hybrid Framework with Ordinal Loss
        m_hybrid = GatedAttentionFusionModel(
            deep_dim=128, radiomic_dim=len(expected_48), hidden_dim=64, num_classes=4
        ).to(device)
        opt_hybrid = torch.optim.AdamW(m_hybrid.parameters(), lr=1e-3, weight_decay=1e-4)
        ordinal_crit = OrdinalHeckmattLoss(num_classes=4, ordinal_weight=0.6, class_weights=class_weights_t)

        m_hybrid.train()
        for epoch in range(75):
            opt_hybrid.zero_grad()
            out = m_hybrid(t_deep_tr, t_rad_tr)
            loss_cls = ordinal_crit(out['logits'], t_y_tr)
            # Add ordinal regression penalty
            loss_ord = nn.functional.mse_loss(out['ordinal_score'], t_y_tr.float() + 1.0)
            total_loss = loss_cls + 0.4 * loss_ord
            total_loss.backward()
            opt_hybrid.step()

        m_hybrid.eval()
        with torch.no_grad():
            out_hybrid = m_hybrid(t_deep_va, t_rad_va)
            p_hybrid = out_hybrid['probabilities'].cpu().numpy()
            y_pred_hybrid = p_hybrid.argmax(axis=1)
        
        metrics_hybrid = compute_classification_metrics(y_val, y_pred_hybrid, p_hybrid)
        results['Proposed_Gated_Attention_Hybrid'].append(metrics_hybrid)

        print(f"  XGBoost Accuracy: {results['Baseline_XGBoost_Radiomics'][-1]['accuracy']:.3f}, QWK: {results['Baseline_XGBoost_Radiomics'][-1]['quadratic_weighted_kappa']:.3f}")
        print(f"  Deep Only Accuracy: {results['Deep_ResNet_Only'][-1]['accuracy']:.3f}, QWK: {results['Deep_ResNet_Only'][-1]['quadratic_weighted_kappa']:.3f}")
        print(f"  Early Fusion Accuracy: {results['Early_Fusion'][-1]['accuracy']:.3f}, QWK: {results['Early_Fusion'][-1]['quadratic_weighted_kappa']:.3f}")
        print(f"  Proposed Hybrid Accuracy: {metrics_hybrid['accuracy']:.3f}, QWK: {metrics_hybrid['quadratic_weighted_kappa']:.3f}, Ordinal MAE: {metrics_hybrid['ordinal_mae']:.3f}")

        if metrics_hybrid['quadratic_weighted_kappa'] > best_qwk:
            best_qwk = metrics_hybrid['quadratic_weighted_kappa']
            best_hybrid_weights = m_hybrid.state_dict()
            # Save best model and scaler
            torch.save(m_hybrid.state_dict(), 'data/fshd/gated_hybrid_best.pth')
            joblib.dump(scaler, 'data/fshd/radiomic_scaler.pkl')

        fold += 1

    # 4. Summary Table of Benchmark
    summary = {}
    print("\n" + "=" * 80)
    print("FINAL 5-FOLD PATIENT-LEVEL CROSS-VALIDATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model Architecture':<35} | {'Accuracy':<10} | {'Balanced Acc':<12} | {'Macro F1':<10} | {'QWK (Kappa)':<12} | {'Ord MAE':<8}")
    print("-" * 85)

    for mname, fold_metrics in results.items():
        acc_mean = np.mean([f['accuracy'] for f in fold_metrics])
        acc_std = np.std([f['accuracy'] for f in fold_metrics])
        bacc_mean = np.mean([f['balanced_accuracy'] for f in fold_metrics])
        bacc_std = np.std([f['balanced_accuracy'] for f in fold_metrics])
        f1_mean = np.mean([f['macro_f1'] for f in fold_metrics])
        f1_std = np.std([f['macro_f1'] for f in fold_metrics])
        qwk_mean = np.mean([f['quadratic_weighted_kappa'] for f in fold_metrics])
        qwk_std = np.std([f['quadratic_weighted_kappa'] for f in fold_metrics])
        mae_mean = np.mean([f['ordinal_mae'] for f in fold_metrics])
        mae_std = np.std([f['ordinal_mae'] for f in fold_metrics])

        summary[mname] = {
            'accuracy': f"{acc_mean:.3f} ± {acc_std:.3f}",
            'balanced_accuracy': f"{bacc_mean:.3f} ± {bacc_std:.3f}",
            'macro_f1': f"{f1_mean:.3f} ± {f1_std:.3f}",
            'quadratic_weighted_kappa': f"{qwk_mean:.3f} ± {qwk_std:.3f}",
            'ordinal_mae': f"{mae_mean:.3f} ± {mae_std:.3f}",
            'raw_folds': fold_metrics
        }
        print(f"{mname:<35} | {acc_mean:.3f}±{acc_std:.2f}  | {bacc_mean:.3f}±{bacc_std:.2f}     | {f1_mean:.3f}±{f1_std:.2f}  | {qwk_mean:.3f}±{qwk_std:.2f}   | {mae_mean:.2f}±{mae_std:.2f}")

    # 5. Train Attention U-Net Segmentation Module
    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION OF DEEP ATTENTION U-NET SEGMENTATION MODULE")
    print("=" * 70)
    
    # Select 100 representative scans for segmentation training & held-out test
    import cv2
    seg_samples = df_valid.sample(min(120, len(df_valid)), random_state=42)
    split_pt = int(0.8 * len(seg_samples))
    train_seg = seg_samples.iloc[:split_pt]
    test_seg = seg_samples.iloc[split_pt:]

    print(f"Segmentation splits: Train={len(train_seg)}, Held-out Test={len(test_seg)}")

    unet_model = AttentionUNet(in_channels=1, out_channels=1, init_features=16).to(device)
    seg_crit = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)
    seg_opt = torch.optim.AdamW(unet_model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train for 20 epochs
    unet_model.train()
    t_seg_0 = time.time()
    for ep in range(1, 21):
        losses = []
        for _, s_row in train_seg.iterrows():
            im = cv2.imread(os.path.join('data/fshd/images', s_row['filename']), cv2.IMREAD_GRAYSCALE)
            mk = cv2.imread(os.path.join('data/fshd/masks/masks', s_row['filename']), cv2.IMREAD_GRAYSCALE)
            if im is None or mk is None:
                continue
            im_t = torch.tensor(cv2.resize(im, (256, 256)) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            mk_t = torch.tensor((cv2.resize(mk, (256, 256)) > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            seg_opt.zero_grad()
            pred = unet_model(im_t)
            l = seg_crit(pred, mk_t)
            l.backward()
            seg_opt.step()
            losses.append(l.item())
        if ep % 5 == 0 or ep == 20:
            print(f"Epoch {ep}/20 - Mean Loss: {np.mean(losses):.4f}")

    print(f"Segmentation training finished in {time.time()-t_seg_0:.2f}s!")
    torch.save(unet_model.state_dict(), 'data/fshd/attention_unet_best.pth')

    # Evaluate on held-out test scans
    unet_model.eval()
    test_seg_metrics = []
    with torch.no_grad():
        for _, s_row in test_seg.iterrows():
            im = cv2.imread(os.path.join('data/fshd/images', s_row['filename']), cv2.IMREAD_GRAYSCALE)
            mk = cv2.imread(os.path.join('data/fshd/masks/masks', s_row['filename']), cv2.IMREAD_GRAYSCALE)
            if im is None or mk is None:
                continue
            im_t = torch.tensor(cv2.resize(im, (256, 256)) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred_logits = unet_model(im_t)
            pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
            pred_mask = cv2.resize((pred_prob > 0.5).astype(np.uint8), (512, 512))

            m = compute_segmentation_metrics(pred_mask, mk)
            test_seg_metrics.append(m)

    mean_dice = np.mean([m['dice'] for m in test_seg_metrics])
    mean_iou = np.mean([m['iou'] for m in test_seg_metrics])
    mean_prec = np.mean([m['precision'] for m in test_seg_metrics])
    mean_rec = np.mean([m['recall'] for m in test_seg_metrics])

    print("\nHeld-out Test Segmentation Performance:")
    print(f"  Dice Similarity Coefficient (DSC): {mean_dice:.4f} ({mean_dice*100:.1f}%)")
    print(f"  Intersection over Union (IoU):     {mean_iou:.4f} ({mean_iou*100:.1f}%)")
    print(f"  Precision:                         {mean_prec:.4f}")
    print(f"  Recall (Sensitivity):              {mean_rec:.4f}")

    summary['Segmentation_Attention_UNet'] = {
        'test_dice': float(mean_dice),
        'test_iou': float(mean_iou),
        'test_precision': float(mean_prec),
        'test_recall': float(mean_rec),
        'num_test_scans': len(test_seg_metrics)
    }

    # 6. Save benchmark results to JSON
    with open('data/fshd/benchmark_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nBenchmark results successfully exported to data/fshd/benchmark_results.json!")

if __name__ == '__main__':
    main()

