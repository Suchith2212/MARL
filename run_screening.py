"""
Screening script: Check 1 (comm vs no-comm) and Check 2 (precision screen).

Stage 2 (dynamic) frozen endpoints:
  R_MIN = -15.95  (scripted no-info heuristic)
  R_MAX =  -7.68  (scripted oracle)
  Gate 0 threshold: score >= 0.50 (return >= -11.82)

Usage:
  python run_screening.py --check 1          # Check 1 only
  python run_screening.py --check 2          # Check 2 only
  python run_screening.py --check all        # Both in order

Check 1 criterion: mean(Δ_seed) >= 0.10 AND CI_low > 0.0 (paired t, N=5)
Check 2 criterion: significance test for linear trend across W={1,2,4,8};
                   null result is valid and must be reported as-is.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.vec_mappo import VecMAPPOTrainer

# ── Frozen endpoints ──────────────────────────────────────────────────────────
STAGE  = 'dynamic'
R_MIN  = -15.95
R_MAX  =  -7.68

# ── Run config ────────────────────────────────────────────────────────────────
SEEDS         = [42, 43, 44, 45, 46]
BUDGET        = 2_000_000          # 2M steps per run; extend if curves not flat
B             = 64                 # envs per rollout
T             = 25                 # steps per rollout
STEPS_PER_UPD = B * T             # 1,600
TOTAL_UPDATES = (BUDGET + STEPS_PER_UPD - 1) // STEPS_PER_UPD

WIDTHS        = [1, 2, 4, 8]
CHECK1_MODES  = ['no_comm', 'full_comm']   # full_comm uses fixed_bw=8
CHECK1_THRESHOLD = 0.10

RESULTS_DIR   = 'results/screening'
FIG_DIR       = 'results/figures'
CKPT_DIR      = 'checkpoints'


def normalised_score(ret):
    return (ret - R_MIN) / (R_MAX - R_MIN)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(trainer, num_eval=1000, eval_seed=99999):
    env = BatchedTrackingEnv(num_envs=num_eval, max_cycles=T,
                             stage=STAGE, full_obs=False)
    obs, state = env.reset(seed=eval_seed)
    total_r = torch.zeros(num_eval)
    with torch.no_grad():
        for _ in range(T):
            t_est, _, _ = trainer.act(obs, deterministic=True)
            obs, state, r, done, _ = env.step_with_target_estimates(t_est)
            total_r += r
    mean_r = float(total_r.mean())
    std_r  = float(total_r.std())
    return mean_r, std_r


# ── Single-seed training worker ───────────────────────────────────────────────
def train_seed(args):
    """Worker function: train one (mode, fixed_bw, seed) run."""
    mode, fixed_bw, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"{mode}_bw{fixed_bw}_seed{seed}" if mode == 'full_comm' else f"{mode}_seed{seed}"
    ckpt_path = os.path.join(CKPT_DIR, f"screening_{tag}.pt")
    result_path = os.path.join(RESULTS_DIR, f"{tag}.json")

    # Fast-path: return cached result if run already finished
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as f:
                rec = json.load(f)
            if 'eval_records' in rec and 'final' in rec['eval_records']:
                sc = rec['eval_records']['final']['score']
                print(f"[{tag}] Already completed with score {sc:.4f}, using cached record.", flush=True)
                return rec
        except Exception:
            pass

    # Resume from checkpoint if available
    start_update = 1
    learning_curve = []
    eval_records   = {}
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_update   = ckpt['update'] + 1
            learning_curve = ckpt.get('learning_curve', [])
            eval_records   = ckpt.get('eval_records', {})
            trainer        = ckpt['trainer']
            print(f"[{tag}] Resumed from update {ckpt['update']} "
                  f"({ckpt['update']*STEPS_PER_UPD:,} steps)")
        except Exception as e:
            print(f"[{tag}] Checkpoint load failed ({e}), starting fresh")
            trainer = VecMAPPOTrainer(mode=mode, fixed_bw=fixed_bw)
    else:
        trainer = VecMAPPOTrainer(mode=mode, fixed_bw=fixed_bw)

    env = BatchedTrackingEnv(num_envs=B, max_cycles=T,
                             stage=STAGE, full_obs=False)
    t0 = time.time()

    # Milestone every 500k steps
    milestones = [500_000, 1_000_000, 1_500_000, 2_000_000]

    for update in range(start_update, TOTAL_UPDATES + 1):
        obs, state = env.reset(seed=seed * 10_000 + update)

        buf_states, buf_scout_obs = [], []
        buf_rover1_obs, buf_rover2_obs = [], []
        buf_p1, buf_p2, buf_lp1, buf_lp2 = [], [], [], []
        buf_rewards, buf_dones = [], []

        with torch.no_grad():
            for t_step in range(T):
                t_est, (p1, p2), (lp1, lp2) = trainer.act(obs, deterministic=False)
                next_obs, next_state, reward, done, _ = env.step_with_target_estimates(t_est)

                buf_states.append(state)
                buf_scout_obs.append(obs['agent_0'])
                buf_rover1_obs.append(obs['agent_1'][:, 0:10])
                buf_rover2_obs.append(obs['agent_2'][:, 0:10])
                buf_p1.append(p1); buf_p2.append(p2)
                buf_lp1.append(lp1); buf_lp2.append(lp2)
                buf_rewards.append(reward)
                buf_dones.append(done.float())

                obs, state = next_obs, next_state

        rollouts = {
            'states'    : torch.stack(buf_states),
            'scout_obs' : torch.stack(buf_scout_obs),
            'rover1_obs': torch.stack(buf_rover1_obs),
            'rover2_obs': torch.stack(buf_rover2_obs),
            'p1_hat'    : torch.stack(buf_p1),
            'p2_hat'    : torch.stack(buf_p2),
            'lp1'       : torch.stack(buf_lp1),
            'lp2'       : torch.stack(buf_lp2),
            'rewards'   : torch.stack(buf_rewards),
            'dones'     : torch.stack(buf_dones),
        }
        trainer.train_step(rollouts)

        curr_steps = update * STEPS_PER_UPD

        # Milestone evaluation
        for m in milestones:
            if (update - 1) * STEPS_PER_UPD < m <= curr_steps:
                mean_r, std_r = evaluate(trainer)
                score = normalised_score(mean_r)
                eval_records[m] = {
                    'return': mean_r, 'std': std_r,
                    'score': score, 'steps': curr_steps,
                }
                elapsed = time.time() - t0
                print(f"  [{tag} | {m:,}] ({elapsed:.0f}s) "
                      f"Return={mean_r:.3f}±{std_r:.2f}  Score={score:.4f}", flush=True)

        # Learning curve every 50 updates (~80k steps)
        if update % 50 == 0:
            mean_r, _ = evaluate(trainer, num_eval=200, eval_seed=update)
            learning_curve.append((curr_steps, mean_r, normalised_score(mean_r)))

        # Save checkpoint every 500 updates (~800k steps)
        if update % 500 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({
                'update'        : update,
                'trainer'       : trainer,
                'learning_curve': learning_curve,
                'eval_records'  : eval_records,
            }, ckpt_path)

    # Final eval
    mean_r, std_r = evaluate(trainer)
    score = normalised_score(mean_r)
    eval_records['final'] = {'return': mean_r, 'std': std_r, 'score': score}
    print(f"  [{tag}] FINAL: Return={mean_r:.3f}  Score={score:.4f}", flush=True)

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'update': TOTAL_UPDATES, 'trainer': trainer,
                'learning_curve': learning_curve, 'eval_records': eval_records}, ckpt_path)

    # Save result JSON
    record = {
        'mode'          : mode,
        'fixed_bw'      : fixed_bw,
        'seed'          : seed,
        'stage'         : STAGE,
        'budget'        : BUDGET,
        'R_MIN'         : R_MIN,
        'R_MAX'         : R_MAX,
        'eval_records'  : {str(k): v for k, v in eval_records.items()},
        'learning_curve': learning_curve,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


# ── Check 1: comm vs no-comm ──────────────────────────────────────────────────
def run_check1():
    print('\n' + '=' * 70)
    print('CHECK 1: Full-comm (b=8) vs No-comm — Stage 2 Dynamic')
    print(f'  Seeds={SEEDS}, Budget={BUDGET:,} steps')
    print('=' * 70)

    # Build work items
    work = [(m, 8 if m == 'full_comm' else 0, s)
            for m in CHECK1_MODES for s in SEEDS]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(work)) as ex:
        results = list(ex.map(train_seed, work))
    print(f'\nCheck 1 training done in {time.time()-t0:.0f}s')

    # Collect per-seed final scores
    no_comm_scores  = {}
    full_comm_scores = {}
    for r in results:
        s = r['seed']
        sc = r['eval_records']['final']['score']
        if r['mode'] == 'no_comm':
            no_comm_scores[s] = sc
        else:
            full_comm_scores[s] = sc

    deltas = [full_comm_scores[s] - no_comm_scores[s] for s in SEEDS]
    mean_d = float(np.mean(deltas))
    se_d   = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
    t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
    ci_low  = mean_d - 2.776 * se_d   # t_{4, 0.025} = 2.776 for N=5
    ci_high = mean_d + 2.776 * se_d

    print('\n' + '=' * 70)
    print('CHECK 1 RESULTS')
    print('=' * 70)
    print(f'  No-comm scores  : {[round(no_comm_scores[s], 4) for s in SEEDS]}')
    print(f'  Full-comm scores: {[round(full_comm_scores[s], 4) for s in SEEDS]}')
    print(f'  Δ per seed      : {[round(d, 4) for d in deltas]}')
    print(f'  Mean Δ          : {mean_d:.4f}')
    print(f'  95% CI          : [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'  t={t_stat:.3f}, p={p_val:.4f} (one-sample t, H0: Δ=0)')
    criterion = mean_d >= CHECK1_THRESHOLD and ci_low > 0.0
    print(f'  Criterion (mean≥0.10 AND CI_low>0): {"PASSED" if criterion else "NOT PASSED"}')
    print('=' * 70)

    summary = {
        'check': 1,
        'seeds': SEEDS,
        'no_comm_scores' : no_comm_scores,
        'full_comm_scores': full_comm_scores,
        'deltas'  : deltas,
        'mean_delta': mean_d,
        'ci_95'   : [ci_low, ci_high],
        't_stat'  : t_stat,
        'p_value' : p_val,
        'criterion_passed': criterion,
    }
    with open(os.path.join(RESULTS_DIR, 'check1_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Check 1 summary → results/screening/check1_summary.json')

    _plot_check1(results, no_comm_scores, full_comm_scores)
    return criterion, summary


# ── Check 2: precision screen ─────────────────────────────────────────────────
def run_check2():
    print('\n' + '=' * 70)
    print('CHECK 2: Precision screen — W = {1, 2, 4, 8}')
    print(f'  Seeds={SEEDS}, Budget={BUDGET:,} steps')
    print('=' * 70)

    work = [('full_comm', bw, s) for bw in WIDTHS for s in SEEDS]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(work)) as ex:
        results = list(ex.map(train_seed, work))
    print(f'\nCheck 2 training done in {time.time()-t0:.0f}s')

    # Collect scores by width
    scores_by_bw = {bw: [] for bw in WIDTHS}
    for r in results:
        bw = r['fixed_bw']
        sc = r['eval_records']['final']['score']
        scores_by_bw[bw].append(sc)

    print('\n' + '=' * 70)
    print('CHECK 2 RESULTS')
    print('=' * 70)
    means, stds = {}, {}
    for bw in WIDTHS:
        sc = scores_by_bw[bw]
        means[bw] = float(np.mean(sc))
        stds[bw]  = float(np.std(sc, ddof=1))
        print(f'  b={bw}: scores={[round(s, 4) for s in sc]}  '
              f'mean={means[bw]:.4f}  std={stds[bw]:.4f}')

    # Pairwise flatness
    bw_list  = WIDTHS
    all_diffs = [(bw_list[i], bw_list[j],
                  abs(means[bw_list[i]] - means[bw_list[j]]))
                 for i in range(len(bw_list)) for j in range(i+1, len(bw_list))]
    max_diff = max(d for _,_,d in all_diffs)
    print(f'\n  Max pairwise |mean diff|: {max_diff:.4f}  '
          f'(flatness threshold <0.10)')

    # Trend test: Spearman rank correlation of mean score vs log2(bw)
    bw_log2  = np.log2(WIDTHS)
    mean_arr = np.array([means[bw] for bw in WIDTHS])
    rho, p_spearman = stats.spearmanr(bw_log2, mean_arr)
    print(f'  Spearman ρ(log2(bw), score): {rho:.4f}, p={p_spearman:.4f}')

    # Paired t: b=8 vs b=1
    d_pair = np.array(scores_by_bw[8]) - np.array(scores_by_bw[1])
    t_pair, p_pair = stats.ttest_1samp(d_pair, 0.0)
    ci_low  = float(np.mean(d_pair)) - 2.776 * float(np.std(d_pair, ddof=1) / np.sqrt(5))
    ci_high = float(np.mean(d_pair)) + 2.776 * float(np.std(d_pair, ddof=1) / np.sqrt(5))
    print(f'  Paired t (b=8 vs b=1): mean Δ={np.mean(d_pair):.4f}, '
          f'95% CI=[{ci_low:.4f},{ci_high:.4f}], t={t_pair:.3f}, p={p_pair:.4f}')

    is_flat    = bool(max_diff < 0.10)
    has_trend  = bool(p_spearman < 0.05)
    print(f'\n  Flat across widths (max|Δ|<0.10): {"YES" if is_flat else "NO"}')
    print(f'  Significant trend  (Spearman p<0.05): {"YES" if has_trend else "NO"}')
    print('  NOTE: null/flat result is a valid scientific finding — do not suppress.')
    print('=' * 70)

    summary = {
        'check'       : 2,
        'widths'      : WIDTHS,
        'seeds'       : SEEDS,
        'scores_by_bw': {str(k): [float(x) for x in v] for k, v in scores_by_bw.items()},
        'means'       : {str(k): float(v) for k, v in means.items()},
        'stds'        : {str(k): float(v) for k, v in stds.items()},
        'max_pairwise_diff': float(max_diff),
        'is_flat'     : bool(is_flat),
        'spearman_rho': float(rho),
        'spearman_p'  : float(p_spearman),
        'has_trend'   : bool(has_trend),
        'paired_t_b8_b1': {'mean_delta': float(np.mean(d_pair)), 'ci_95': [float(ci_low), float(ci_high)],
                            't': float(t_pair), 'p': float(p_pair)},
    }
    with open(os.path.join(RESULTS_DIR, 'check2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Check 2 summary → results/screening/check2_summary.json')

    _plot_check2(means, stds, scores_by_bw)
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────
def _plot_check1(results, no_comm_scores, full_comm_scores):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Learning curves
        ax = axes[0]
        for r in results:
            xs = [pt[0] for pt in r['learning_curve']]
            ys = [pt[2] for pt in r['learning_curve']]  # normalised score
            lbl = f"{r['mode']} s={r['seed']}"
            ls  = '-' if r['mode'] == 'full_comm' else '--'
            ax.plot(xs, ys, ls=ls, lw=1.2, alpha=0.7, label=lbl)
        ax.axhline(0, color='purple', ls=':', lw=1.5, label='Heuristic (0.0)')
        ax.axhline(1, color='darkgreen', ls=':', lw=1.5, label='Oracle (1.0)')
        ax.set_xlabel('Steps'); ax.set_ylabel('Normalised score')
        ax.set_title('Check 1 — Learning Curves (all seeds)')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # Final score comparison
        ax = axes[1]
        sc_nc = [no_comm_scores[s] for s in SEEDS]
        sc_fc = [full_comm_scores[s] for s in SEEDS]
        x = np.arange(len(SEEDS))
        ax.bar(x - 0.2, sc_nc, 0.35, label='No-comm', color='steelblue', alpha=0.8)
        ax.bar(x + 0.2, sc_fc, 0.35, label='Full-comm b=8', color='coral', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f's={s}' for s in SEEDS])
        ax.set_ylabel('Final normalised score')
        ax.set_title('Check 1 — Final Scores per Seed')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'check1_results.png'), dpi=300)
        plt.close()
        print('Plot → results/figures/check1_results.png')
    except Exception as e:
        print(f'Plot failed: {e}')


def _plot_check2(means, stds, scores_by_bw):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Score vs bit-width
        ax = axes[0]
        bw_arr = np.array(WIDTHS)
        m_arr  = np.array([means[b] for b in WIDTHS])
        s_arr  = np.array([stds[b]  for b in WIDTHS])
        ax.errorbar(bw_arr, m_arr, yerr=1.96*s_arr/np.sqrt(5),
                    fmt='o-', color='teal', capsize=5, lw=2)
        for bw in WIDTHS:
            jitter = np.random.uniform(-0.15, 0.15, 5)
            ax.scatter([bw]*5 + jitter, scores_by_bw[bw],
                       color='gray', s=20, alpha=0.6, zorder=3)
        ax.set_xlabel('Bit-width b'); ax.set_ylabel('Normalised score')
        ax.set_xticks(WIDTHS)
        ax.set_title('Check 2 — Score vs Bit-width\n(mean ± 95% CI, individual seeds)')
        ax.grid(alpha=0.3)

        # Box-plot
        ax = axes[1]
        ax.boxplot([scores_by_bw[b] for b in WIDTHS],
                   labels=[f'b={b}' for b in WIDTHS], notch=False)
        ax.set_ylabel('Normalised score')
        ax.set_title('Check 2 — Score Distribution per Width')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'check2_results.png'), dpi=300)
        plt.close()
        print('Plot → results/figures/check2_results.png')
    except Exception as e:
        print(f'Plot failed: {e}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', choices=['1', '2', 'all'], default='all')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if args.check in ('1', 'all'):
        passed, c1 = run_check1()
        if args.check == 'all' and not passed:
            print('\nCheck 1 NOT PASSED — stopping. Do not proceed to Check 2.')
            sys.exit(0)

    if args.check in ('2', 'all'):
        run_check2()
