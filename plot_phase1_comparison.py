import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.envs.po_mpe import PartiallyObservableSpreadEnv
from scripts.run_phase1_screen import train_seed

def main():
    seeds = [42, 43, 44, 45, 46]
    test_seeds = [1000 + i * 17 for i in range(20)]
    
    # Scripted baselines from eval_scripted_phase1:
    oracle_val = 31.08
    heur_val = -12.42
    random_val = -14.00
    
    # Collect curves for no-comm, full-comm (b=8), full-obs
    # Train 5 seeds each (fast in parallel) or run sequential
    print("Collecting learning curves across 5 seeds for Phase 1 Clue-Nav task...")
    from concurrent.futures import ProcessPoolExecutor
    
    def get_curves(mode, fixed_bw, full_obs):
        curves = []
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(train_seed, mode, fixed_bw, s, test_seeds, full_obs, 25, 12, -25.0, 0.0)
                for s in seeds
            ]
            for fut in futures:
                res = fut.result()
                curves.append(res['learning_curve'])
        return curves

    print("  Running full-obs...")
    fo_curves = get_curves('no_comm', 0, True)
    print("  Running no-comm...")
    nc_curves = get_curves('no_comm', 0, False)
    print("  Running full-comm (b=8)...")
    fc_curves = get_curves('full_comm_fixed', 8, False)
    
    plt.figure(figsize=(10, 6))
    updates = np.arange(1, 26)
    
    for name, c_list, col in [('Full-Observation', fo_curves, 'green'),
                              ('Full-Comm (b=8)', fc_curves, 'royalblue'),
                              ('No-Comm', nc_curves, 'crimson')]:
        m = np.mean(c_list, axis=0)
        s = np.std(c_list, axis=0)
        plt.plot(updates, m, label=f"{name} (final: {m[-1]:.2f})", color=col, linewidth=2)
        plt.fill_between(updates, m - s, m + s, alpha=0.15, color=col)
        
    plt.axhline(oracle_val, color='darkgreen', linestyle='-', linewidth=1.5, label=f'Scripted Oracle ({oracle_val:.1f})')
    plt.axhline(heur_val, color='purple', linestyle='--', linewidth=1.5, label=f'Scripted No-Info Heuristic ({heur_val:.1f})')
    plt.axhline(random_val, color='gray', linestyle=':', linewidth=1.5, label=f'Random Policy ({random_val:.1f})')
    
    plt.xlabel("PPO Update (12 eps / update)", fontsize=11)
    plt.ylabel("Raw Episode Return", fontsize=11)
    plt.title("Phase 1 Target Clue Exchange: Learning Curves & Baselines (5 Seeds)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/phase1_with_scripted_baselines.png", dpi=300)
    plt.close()
    print("Saved plot to: results/figures/phase1_with_scripted_baselines.png")

if __name__ == '__main__':
    main()
