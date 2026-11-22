import os
import json
import glob

RESULTS_DIR = "results/phase3_full"
files = glob.glob(os.path.join(RESULTS_DIR, "learned_precision_*.json"))
print(f"Total completed Phase 3 runs: {len(files)} / 20")
for f in sorted(files):
    tag = os.path.basename(f).replace(".json", "")
    try:
        with open(f, 'r') as fp:
            d = json.load(fp)
            ev = d.get('eval', {})
            ret = ev.get('return_mean', 0.0)
            cost = ev.get('cost_mean', 0.0)
            bgt = d.get('budget', 0)
            err = ev.get('cost_error', 0.0)
            nu = ev.get('nu', 0.0)
            wf = ev.get('width_fracs', {})
            w_str = f"1:{wf.get('1',0):.1f} 2:{wf.get('2',0):.1f} 4:{wf.get('4',0):.1f} 8:{wf.get('8',0):.1f}"
            print(f"{tag:<32}: Ret={ret:6.2f}, Cost={cost:5.1f}/{bgt} (Err={err:+.1%}), nu={nu:.3f}, widths=[{w_str}]")
    except Exception as e:
        pass
