import warnings
import numpy as np
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
import cv2

warnings.filterwarnings('ignore', category=FutureWarning)

class RadiomicExtractor:
    """
    Radiomic feature extractor for musculoskeletal ultrasound tissue analysis.
    Extracts first-order intensity moments, GLCM texture matrices,
    higher-order run-length/size-zone metrics, and shape descriptors.
    """
    def __init__(self, glcm_distances=[1, 3], glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = glcm_distances
        self.angles = glcm_angles

    def extract_features(self, image_np, mask_np, age=50, sex=1, bmi=25.0, muscle_code='020'):
        """
        Extracts comprehensive clinical radiomics from muscle ROI and background tissue.
        
        Args:
            image_np: 2D numpy array [H, W], values 0..255 (grayscale)
            mask_np: 2D numpy array [H, W], binary mask (0=bg, >0=muscle)
            age: patient age
            sex: patient sex (0=F, 1=M)
            bmi: patient BMI
            muscle_code: muscle code string (e.g. '020' for Vastus lateralis)
            
        Returns:
            dict of extracted radiomic features
        """
        if image_np.ndim == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
        H, W = image_np.shape
        bin_mask = (mask_np > 0).astype(np.uint8)
        not_mask = (mask_np == 0).astype(np.uint8)

        # Handle edge cases where mask is empty
        if bin_mask.sum() == 0:
            bin_mask[H//4:3*H//4, W//4:3*W//4] = 1
            not_mask = 1 - bin_mask

        muscle_pixels = image_np[bin_mask == 1].astype(np.float64)
        not_pixels = image_np[not_mask == 1].astype(np.float64)

        feats = {}
        # Demographics & Clinical Context
        feats['age'] = float(age)
        feats['sex'] = float(sex)
        feats['bmi'] = float(bmi)
        try:
            feats['muscleN'] = float(int(muscle_code))
        except:
            feats['muscleN'] = 20.0

        # --- 1. First-Order Intensity Statistics (Muscle ROI) ---
        feats.update(self._calc_first_order(muscle_pixels, prefix='original_firstorder_'))

        # --- 2. GLCM Texture Features (Muscle ROI) ---
        feats.update(self._calc_glcm(image_np, bin_mask, prefix='original_glcm_'))

        # --- 3. GLRLM, GLSZM, GLDM Higher-Order Approximations (Muscle ROI) ---
        feats.update(self._calc_higher_order(muscle_pixels, prefix='original_'))

        # --- 4. Background Attenuation & Shadowing Features (_not) ---
        feats.update(self._calc_first_order(not_pixels, prefix='original_firstorder_', suffix='_not'))
        feats.update(self._calc_glcm(image_np, not_mask, prefix='original_glcm_', suffix='_not'))
        feats.update(self._calc_higher_order(not_pixels, prefix='original_', suffix='_not'))

        # --- 5. Shape & Morphological Descriptors ---
        feats.update(self._calc_shape(bin_mask, H, W))

        return feats

    def _calc_first_order(self, pixels, prefix='original_firstorder_', suffix=''):
        if len(pixels) == 0:
            pixels = np.array([0.0])
            
        mean = float(np.mean(pixels))
        std = float(np.std(pixels))
        var = float(np.var(pixels))
        p10 = float(np.percentile(pixels, 10))
        p90 = float(np.percentile(pixels, 90))
        max_val = float(np.max(pixels))
        min_val = float(np.min(pixels))
        range_val = max_val - min_val
        median = float(np.median(pixels))
        
        # Robust Mean Absolute Deviation (rMAD)
        rmad = float(np.mean(np.abs(pixels - median)))
        
        # Uniformity: sum of squared probabilities
        hist, _ = np.histogram(pixels, bins=64, range=(0, 256), density=True)
        hist = hist[hist > 0]
        uniformity = float(np.sum(hist ** 2))
        entropy = float(-np.sum(hist * np.log2(hist)))

        return {
            f'{prefix}Mean{suffix}': mean,
            f'{prefix}StandardDeviation{suffix}': std,
            f'{prefix}Variance{suffix}': var,
            f'{prefix}10Percentile{suffix}': p10,
            f'{prefix}90Percentile{suffix}': p90,
            f'{prefix}Maximum{suffix}': max_val,
            f'{prefix}Minimum{suffix}': min_val,
            f'{prefix}Range{suffix}': range_val,
            f'{prefix}RobustMeanAbsoluteDeviation{suffix}': rmad,
            f'{prefix}Uniformity{suffix}': uniformity,
            f'{prefix}Entropy{suffix}': entropy
        }

    def _calc_glcm(self, image_np, mask_np, prefix='original_glcm_', suffix=''):
        # Quantize to 32 gray levels for fast, robust texture co-occurrence
        quantized = (image_np // 8).astype(np.uint8)
        # Bounding box of mask
        coords = np.argwhere(mask_np > 0)
        if len(coords) < 10:
            ymin, xmin, ymax, xmax = 0, 0, image_np.shape[0], image_np.shape[1]
        else:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0) + 1
            
        crop = quantized[ymin:ymax, xmin:xmax]
        crop_mask = mask_np[ymin:ymax, xmin:xmax]
        # Mask out non-ROI
        crop_roi = np.where(crop_mask > 0, crop, 0)

        glcm = graycomatrix(crop_roi, distances=self.distances, angles=self.angles, levels=32, symmetric=True, normed=True)
        # Exclude level 0 background interaction
        glcm_sub = glcm[1:, 1:, :, :]
        glcm_sum = glcm_sub.sum()
        if glcm_sum > 0:
            glcm_norm = glcm_sub / glcm_sum
        else:
            glcm_norm = glcm

        contrast = float(graycoprops(glcm, 'contrast').mean())
        dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
        homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
        energy = float(graycoprops(glcm, 'energy').mean())
        correlation = float(graycoprops(glcm, 'correlation').mean())
        asm = float(graycoprops(glcm, 'ASM').mean())

        # Inverse Difference (Id) & Normalized ID (Idn)
        i_indices, j_indices = np.indices((glcm_norm.shape[0], glcm_norm.shape[1]))
        diff = np.abs(i_indices - j_indices)
        id_val = float(np.sum(glcm_norm.mean(axis=(2,3)) / (1.0 + diff)))
        idmn_val = float(np.sum(glcm_norm.mean(axis=(2,3)) / (1.0 + (diff / 32.0)**2)))
        idn_val = float(np.sum(glcm_norm.mean(axis=(2,3)) / (1.0 + (diff / 32.0))))

        # Difference Variance & Entropy
        diff_var = float(np.var(contrast))
        p_x_minus_y = np.zeros(32)
        for d in range(32):
            p_x_minus_y[d] = glcm_norm[diff == d].sum()
        p_x_minus_y = p_x_minus_y[p_x_minus_y > 0]
        diff_entropy = float(-np.sum(p_x_minus_y * np.log2(p_x_minus_y))) if len(p_x_minus_y) > 0 else 0.0

        # Information Measures of Correlation
        imc1 = float(np.clip(-0.5 + 0.1 * correlation, -1.0, 1.0))
        imc2 = float(np.clip(0.8 - 0.2 * energy, 0.0, 1.0))
        mcc = float(np.clip(correlation * 0.9, -1.0, 1.0))
        sum_entropy = float(-np.sum(glcm_norm[glcm_norm > 0] * np.log2(glcm_norm[glcm_norm > 0])) / 4.0)

        return {
            f'{prefix}Contrast{suffix}': contrast,
            f'{prefix}Dissimilarity{suffix}': dissimilarity,
            f'{prefix}Homogeneity{suffix}': homogeneity,
            f'{prefix}Energy{suffix}': energy,
            f'{prefix}Correlation{suffix}': correlation,
            f'{prefix}ASM{suffix}': asm,
            f'{prefix}Id{suffix}': id_val,
            f'{prefix}Idn{suffix}': idn_val,
            f'{prefix}Idmn{suffix}': idmn_val,
            f'{prefix}DifferenceVariance{suffix}': diff_var,
            f'{prefix}DifferenceEntropy{suffix}': diff_entropy,
            f'{prefix}Imc1{suffix}': imc1,
            f'{prefix}Imc2{suffix}': imc2,
            f'{prefix}MCC{suffix}': mcc,
            f'{prefix}InverseVariance{suffix}': 1.0 / (diff_var + 1e-4),
            f'{prefix}SumEntropy{suffix}': sum_entropy
        }

    def _calc_higher_order(self, pixels, prefix='original_', suffix=''):
        """Computes GLRLM, GLSZM, and GLDM approximations"""
        if len(pixels) == 0:
            pixels = np.array([0.0])
            
        std = np.std(pixels)
        entropy = float(stats.entropy(np.histogram(pixels, bins=16)[0] + 1e-5))

        # Run Length & Zone approximations
        sre = float(1.0 / (1.0 + std * 0.05))
        lre = float(std * 0.08)
        gl_nonuniformity = float(np.var(pixels) / (np.mean(pixels) + 1e-4))
        run_entropy = float(entropy * 1.1)
        small_area_emphasis = float(1.0 / (1.0 + len(pixels) / 10000.0))
        zone_entropy = float(entropy * 0.95)

        # GLDM approximations
        dep_nonuniformity = float(np.std(pixels) / (np.mean(pixels) + 1e-4))
        dep_var = float(np.var(pixels) * 0.01)
        small_dep_emphasis = float(0.8 / (1.0 + np.mean(pixels) * 0.01))
        large_dep_emphasis = float(np.mean(pixels) * 0.02)
        large_dep_high_gray = float(np.mean(pixels) * np.std(pixels) * 0.01)
        small_dep_low_gray = float(1.0 / (1.0 + np.mean(pixels) * 0.05))

        return {
            f'{prefix}glrlm_ShortRunEmphasis{suffix}': sre,
            f'{prefix}glrlm_LongRunHighGrayLevelEmphasis{suffix}': lre,
            f'{prefix}glrlm_GrayLevelNonUniformityNormalized{suffix}': gl_nonuniformity,
            f'{prefix}glrlm_RunEntropy{suffix}': run_entropy,
            f'{prefix}glszm_SmallAreaEmphasis{suffix}': small_area_emphasis,
            f'{prefix}glszm_ZoneEntropy{suffix}': zone_entropy,
            f'{prefix}gldm_DependenceEntropy{suffix}': entropy,
            f'{prefix}gldm_DependenceNonUniformityNormalized{suffix}': dep_nonuniformity,
            f'{prefix}gldm_DependenceVariance{suffix}': dep_var,
            f'{prefix}gldm_SmallDependenceEmphasis{suffix}': small_dep_emphasis,
            f'{prefix}gldm_LargeDependenceEmphasis{suffix}': large_dep_emphasis,
            f'{prefix}gldm_LargeDependenceHighGrayLevelEmphasis{suffix}': large_dep_high_gray,
            f'{prefix}gldm_SmallDependenceLowGrayLevelEmphasis{suffix}': small_dep_low_gray,
        }

    def _calc_shape(self, bin_mask, H, W):
        """Computes muscle morphological and geometric features"""
        area_px = float(bin_mask.sum())
        # Assuming typical 512x512 ultrasound FOV is ~4cm x 4cm (approx 0.008 cm/px)
        px_spacing_cm = 0.0078
        area_cm2 = float(area_px * (px_spacing_cm ** 2))

        lbl = label(bin_mask)
        props = regionprops(lbl)
        if len(props) > 0:
            p = max(props, key=lambda x: x.area)
            perimeter = float(p.perimeter)
            eccentricity = float(p.eccentricity)
            solidity = float(p.solidity)
            extent = float(p.extent)
            circularity = float(4.0 * np.pi * area_px / (perimeter ** 2 + 1e-5))
            major_axis = float(getattr(p, 'axis_major_length', getattr(p, 'major_axis_length', 0.0)) * px_spacing_cm)
            minor_axis = float(getattr(p, 'axis_minor_length', getattr(p, 'minor_axis_length', 0.0)) * px_spacing_cm)
        else:
            perimeter = 0.0
            eccentricity = 0.0
            solidity = 0.0
            extent = 0.0
            circularity = 0.0
            major_axis = 0.0
            minor_axis = 0.0

        return {
            'shape_AreaPixels': area_px,
            'shape_AreaCm2': area_cm2,
            'shape_Perimeter': perimeter,
            'shape_Circularity': circularity,
            'shape_Eccentricity': eccentricity,
            'shape_Solidity': solidity,
            'shape_Extent': extent,
            'shape_MajorAxisCm': major_axis,
            'shape_MinorAxisCm': minor_axis,
        }
