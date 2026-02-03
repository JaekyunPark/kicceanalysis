import pandas as pd
import numpy as np
from scipy import stats

# Mock helper functions
def weighted_mean(values, weights):
    try:
        return np.average(values, weights=weights)
    except:
        return 0

def weighted_std(values, weights):
    try:
        average = np.average(values, weights=weights)
        variance = np.average((values - average)**2, weights=weights)
        n = weights.sum()
        if n > 1:
            variance = variance * n / (n - 1)
        return np.sqrt(variance)
    except:
        return 0

def weighted_anova(groups, weights_list):
    # Simplified placeholder or mock for Levene
    # We need to simulate a case where Levene P > 0.05
    pass

# We will copy the actual implementation of weighted_ttest, weighted_ttest_pooled, weighted_levene from views.py
# to verify logic.

def weighted_ttest(group1, group2, w1, w2):
    n1 = w1.sum()
    n2 = w2.sum()
    m1 = weighted_mean(group1, w1)
    m2 = weighted_mean(group2, w2)
    v1 = weighted_std(group1, w1)**2
    v2 = weighted_std(group2, w2)**2
    
    se = np.sqrt(v1/n1 + v2/n2)
    if se == 0: return None
    t_stat = (m1 - m2) / se
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    p_val = stats.t.sf(np.abs(t_stat), df) * 2
    return t_stat, p_val, df

def weighted_ttest_pooled(group1, group2, w1, w2):
    n1 = w1.sum()
    n2 = w2.sum()
    m1 = weighted_mean(group1, w1)
    m2 = weighted_mean(group2, w2)
    v1 = weighted_std(group1, w1)**2
    v2 = weighted_std(group2, w2)**2
    
    df = n1 + n2 - 2
    sp_sq = ((n1 - 1)*v1 + (n2 - 1)*v2) / df
    se = np.sqrt(sp_sq * (1/n1 + 1/n2))
    if se == 0: return None
    t_stat = (m1 - m2) / se
    p_val = stats.t.sf(np.abs(t_stat), df) * 2
    return t_stat, p_val, df

def weighted_anova(groups, weights_list):
    # Standard One-Way Weighted ANOVA logic
    k = len(groups)
    N = sum(w.sum() for w in weights_list)
    grand_mean = sum(weighted_mean(g, w) * w.sum() for g, w in zip(groups, weights_list)) / N
    
    ss_between = sum(w.sum() * (weighted_mean(g, w) - grand_mean)**2 for g, w in zip(groups, weights_list))
    
    ss_within = 0
    for g, w in zip(groups, weights_list):
        m = weighted_mean(g, w)
        ss_within += np.sum(w * (g - m)**2)
        
    df_between = k - 1
    df_within = N - k
    
    if df_between == 0 or df_within == 0: return None
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    if ms_within == 0: return None
    
    f_stat = ms_between / ms_within
    p_val = stats.f.sf(f_stat, df_between, df_within)
    return f_stat, p_val, df_between, df_within

def weighted_levene(groups, weights_list):
    group_means = []
    for g, w in zip(groups, weights_list):
        group_means.append(weighted_mean(g, w))
        
    deviations = []
    dev_weights = []
    
    for i, (g, w) in enumerate(zip(groups, weights_list)):
        mean = group_means[i]
        dev = np.abs(g - mean)
        deviations.append(dev)
        dev_weights.append(w)
        
    return weighted_anova(deviations, dev_weights)

# Test Case
# Scenario: Two groups with similar variances (Expected: Levene P > 0.05, Student's T used)
np.random.seed(42)
g1 = np.array([10, 11, 12, 11, 10])
w1 = np.array([1.0, 1.2, 0.8, 1.1, 1.0])

g2 = np.array([12, 13, 14, 13, 12]) # shifted mean, similar variance
w2 = np.array([1.0, 0.9, 1.1, 1.0, 1.0])

groups = [pd.Series(g1), pd.Series(g2)]
weights = [pd.Series(w1), pd.Series(w2)]

print("=== Debug Run (Equal Variances) ===")
# 1. Levene
levene_res = weighted_levene(groups, weights)
print(f"Levene Result: {levene_res}") # (F, p, df1, df2)
if levene_res:
    levene_p = levene_res[1]
else:
    levene_p = 1
print(f"Levene P: {levene_p:.4f}")

# 2. Welch T
t_welch, p_welch, df_welch = weighted_ttest(groups[0], groups[1], weights[0], weights[1])
print(f"Welch T: {t_welch:.4f}")

# 3. Pooled T
t_pooled, p_pooled, df_pooled = weighted_ttest_pooled(groups[0], groups[1], weights[0], weights[1])
print(f"Pooled T: {t_pooled:.4f}")

# Selection
if levene_p < 0.05:
    selected = "Welch"
    val = t_welch
else:
    selected = "Student (Pooled)"
    val = t_pooled

print(f"Selected: {selected}, Value: {val:.4f}")


# Scenario 2: Different Variances
print("\n=== Debug Run (Unequal Variances) ===")
g3 = np.array([10, 10, 10, 10, 10]) # Zero variance
w3 = np.array([1, 1, 1, 1, 1])
g4 = np.array([10, 20, 30, 40, 50]) # High variance
w4 = np.array([1, 1, 1, 1, 1])

groups_diff = [pd.Series(g3), pd.Series(g4)]
weights_diff = [pd.Series(w3), pd.Series(w4)]

levene_res_diff = weighted_levene(groups_diff, weights_diff)
print(f"Levene Result: {levene_res_diff}")
if levene_res_diff:
    levene_p_diff = levene_res_diff[1]
else:
    levene_p_diff = 1 # ? logic check
    
print(f"Levene P: {levene_p_diff:.4f}")

t_welch_diff, _, _ = weighted_ttest(groups_diff[0], groups_diff[1], weights_diff[0], weights_diff[1])
t_pooled_diff, _, _ = weighted_ttest_pooled(groups_diff[0], groups_diff[1], weights_diff[0], weights_diff[1])

print(f"Welch T: {t_welch_diff:.4f}")
print(f"Pooled T: {t_pooled_diff:.4f}")

if levene_p_diff < 0.05:
    print(f"Selected: Welch ({t_welch_diff:.4f})")
else:
    print(f"Selected: Student ({t_pooled_diff:.4f})")

print("\n=== Debug Run (Rounding Weights - Scenario 1) ===")
# Round weights to nearest integer
w1_round = w1.round()
w2_round = w2.round()
weights_round = [pd.Series(w1_round), pd.Series(w2_round)]

levene_res_r = weighted_levene(groups, weights_round)
print(f"Levene Result (Rounded): {levene_res_r}")
if levene_res_r:
    print(f"Levene P (Rounded): {levene_res_r[1]:.4f}")

t_welch_r, _, _ = weighted_ttest(groups[0], groups[1], weights_round[0], weights_round[1])
t_pooled_r, _, _ = weighted_ttest_pooled(groups[0], groups[1], weights_round[0], weights_round[1])
print(f"Welch T (Rounded): {t_welch_r:.4f}")
print(f"Pooled T (Rounded): {t_pooled_r:.4f}")

