import json, glob, os

files = glob.glob('results/phase2_full/*.json')
print(f"Total completed runs: {len(files)} / 80")
for f in sorted(files):
    with open(f) as fp:
        d = json.load(fp)
    tag = os.path.basename(f).replace('.json', '')
    e = d['eval']
    err_pct = e['cost_error'] * 100.0
    print(f"{tag:<32}: Ret={e['return_mean']:+.2f}, Cost={e['cost_mean']:5.1f}/{d['budget']} (Err={err_pct:+5.1f}%), nu={e['nu']:.3f}")
