import numpy as np
from scipy.stats import skew, kurtosis

def extract_window_features(win):
    left_s  = win[:, 0:8]
    right_s = win[:, 8:16]
    left_t  = win[:, 16]
    right_t = win[:, 17]
    
    feats = []
    for tot, sens in [(left_t, left_s), (right_t, right_s)]:
        on_ground = (tot > 0.12).astype(int)
        diffs = np.diff(on_ground)
        hs = np.flatnonzero(diffs == 1)
        
        if len(hs) >= 2:
            st = np.diff(hs) / 100.0
            s_mean, s_std = float(np.mean(st)), float(np.std(st))
            s_cv = (s_std / (s_mean + 1e-8)) * 100.0
        else:
            s_mean, s_std, s_cv = 1.1, 0.05, 4.5
            
        stance_ratio = float(np.mean(on_ground))
        heel_f = float(np.mean(sens[:, 0:2]))
        toe_f  = float(np.mean(sens[:, 6:8]))
        ht_ratio = heel_f / (toe_f + 1e-8)
        
        fft_v = np.abs(np.fft.rfft(tot - np.mean(tot)))
        freqs = np.fft.rfftfreq(len(tot), d=0.01)
        w_en = float(np.sum(fft_v[(freqs >= 0.5) & (freqs <= 3.0)] ** 2))
        f_en = float(np.sum(fft_v[(freqs > 3.0) & (freqs <= 8.0)] ** 2))
        freeze_idx = f_en / (w_en + 1e-8)
        
        feats.extend([
            s_mean, s_std, s_cv, stance_ratio, ht_ratio, freeze_idx,
            float(np.max(tot)), float(np.std(tot)), float(skew(tot)), float(kurtosis(tot))
        ])
        feats.extend([float(x) for x in np.mean(sens, axis=0)])
        feats.extend([float(x) for x in np.std(sens, axis=0)])
        
    lr_corr = float(np.corrcoef(left_t, right_t).item(1))
    double_supp = float(np.mean((left_t > 0.12) & (right_t > 0.12)))
    asym_stride = abs(feats.__getitem__(0) - feats.__getitem__(26))
    asym_stance = abs(feats.__getitem__(3) - feats.__getitem__(29))
    
    feats.extend([lr_corr, double_supp, asym_stride, asym_stance])
    return feats
