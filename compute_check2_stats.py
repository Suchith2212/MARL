import json
import numpy as np
from scipy import stats

with open('results/screening/check2_summary.json') as f:
    c2 = json.load(f)

scores = {int(k): np.array(v) for k, v in c2['scores_by_bw'].items()}
widths = [1, 2, 4, 8]
seeds = [42, 43, 44, 45, 46]

print("=== PER-SEED SPEARMAN ===")
rhos = []
for s_idx in range(5):
    s_scores = [scores[w][s_idx] for w in widths]
    r, p = stats.spearmanr(np.log2(widths), s_scores)
    rhos.append(r)
    print(f"Seed {seeds[s_idx]}: scores={[round(x, 4) for x in s_scores]}, rho={r:.4f}")
print(f"Mean per-seed Spearman rho: {np.mean(rhos):.4f}")

# Pooled Spearman across all 20 observations
all_w = np.repeat(np.log2(widths), 5)
all_s = np.concatenate([scores[w] for w in widths])
pooled_rho, pooled_p = stats.spearmanr(all_w, all_s)
print(f"Pooled 20-point Spearman rho: {pooled_rho:.4f}, p={pooled_p:.6f}")

print("\n=== ALL PAIRWISE DIFFERENCES WITH HOLM CORRECTION ===")
pairs = []
for i in range(len(widths)):
    for j in range(i+1, len(widths)):
        w1, w2 = widths[j], widths[i] # higher vs lower
        diff = scores[w1] - scores[w2]
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)
        mean_d = float(np.mean(diff))
        se_d = float(np.std(diff, ddof=1) / np.sqrt(5))
        ci_low = mean_d - 2.776 * se_d
        ci_high = mean_d + 2.776 * se_d
        pairs.append({
            'pair': f"b={w1} vs b={w2}",
            'mean_diff': mean_d,
            'se': se_d,
            'ci_95': (ci_low, ci_high),
            't': float(t_stat),
            'p_raw': float(p_val)
        })

# Sort by raw p-value for Holm-Bonferroni
pairs.sort(key=lambda x: x['p_raw'])
m = len(pairs)
for k, item in enumerate(pairs):
    item['p_holm'] = min(1.0, item['p_raw'] * (m - k))
for k in range(1, m):
    pairs[k]['p_holm'] = max(pairs[k]['p_holm'], pairs[k-1]['p_holm'])

for item in pairs:
    sig = "YES" if item['p_holm'] < 0.05 else "NO"
    print(f"{item['pair']:<14}: mean Δ={item['mean_diff']:+.4f}, 95% CI=[{item['ci_95'][0]:+.4f}, {item['ci_95'][1]:+.4f}], t={item['t']:+.3f}, p_raw={item['p_raw']:.5f}, p_holm={item['p_holm']:.5f}, Holm-sig={sig}")

# Save to json
out = {
    'per_seed_spearman': {str(seeds[i]): float(rhos[i]) for i in range(5)},
    'mean_per_seed_rho': float(np.mean(rhos)),
    'pooled_spearman': {'rho': float(pooled_rho), 'p_value': float(pooled_p)},
    'pairwise_holm': pairs
}
with open('results/screening/check2_pairwise_stats.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/screening/check2_pairwise_stats.json")
