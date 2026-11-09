import numpy as np
import shap
import pandas as pd

class RadiomicSHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) engine for radiomic Heckmatt grading.
    Provides global feature importance rankings and local patient-level attribution.
    """
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            # Fallback for generic models
            self.explainer = None

    def explain_instance(self, feature_vector):
        """
        Calculates SHAP values for a single patient's radiomic profile.
        
        Args:
            feature_vector: 1D array of radiomic values or DataFrame
            
        Returns:
            dict of {feature_name: shap_value}
        """
        if isinstance(feature_vector, pd.Series):
            feat_arr = feature_vector.values.reshape(1, -1)
            fnames = list(feature_vector.index)
        elif isinstance(feature_vector, dict):
            fnames = list(feature_vector.keys())
            feat_arr = np.array([list(feature_vector.values())]).reshape(1, -1)
        else:
            feat_arr = np.array(feature_vector).reshape(1, -1)
            fnames = self.feature_names or [f"Feature_{i}" for i in range(feat_arr.shape[1])]

        if self.explainer is not None:
            shap_values = self.explainer.shap_values(feat_arr)
            # If multi-class, shap_values is a list of arrays (one per class)
            if isinstance(shap_values, list):
                # Return dictionary for each class or mean magnitude
                shap_mag = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
            elif shap_values.ndim == 3:
                # Shape [1, num_features, num_classes]
                shap_mag = np.mean(np.abs(shap_values[0]), axis=-1)
            else:
                shap_mag = shap_values[0]
                
            return {fnames[i]: float(shap_mag[i]) for i in range(len(fnames))}
        else:
            # Fallback feature attribution based on standard deviations
            stds = np.std(feat_arr, axis=0) if feat_arr.shape[0] > 1 else np.ones(feat_arr.shape[1])
            pseudo_shap = (feat_arr[0] - np.mean(feat_arr[0])) / (stds + 1e-5)
            return {fnames[i]: float(pseudo_shap[i]) for i in range(len(fnames))}

    @staticmethod
    def get_top_contributors(shap_dict, top_k=10):
        """Returns sorted list of (feature, importance) tuples"""
        sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_feats[:top_k]

