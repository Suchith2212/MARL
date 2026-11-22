import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor
from scipy import stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from test_smooth_tracking import ContinuousTrackingEnv, get_proportional_action
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from src.utils.metrics import compute_paired_ci_95, save_jsonl

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy_on_test_suite(env, channel, trainer, test_seeds, mode='full_comm', fixed_bw=8):
    returns = []
    c_disc_list = []
    c_bar_list = []
    
    for s in test_seeds:
        obs, state = env.reset(seed=s)
        if channel:
            channel.reset()
        ep_return = 0.0
        
        for t in range(env.max_cycles):
            h_list, raw_msgs, gates, _, bit_widths, _ = trainer.act(obs, deterministic=True)
            
            if mode != 'no_comm' and channel is not None:
                g_eval = [1, 0, 0]
                b_eval = [fixed_bw, 0, 0]
                quantised_msgs = [
                    trainer.quantiser(raw_msgs[i:i+1], bit_width=b_eval[i]).squeeze(0) if g_eval[i] == 1 else torch.zeros(16)
                    for i in range(env.n_agents)
                ]
                delivered, step_cost = channel.transmit_and_deliver(quantised_msgs, g_eval, b_eval)
                actions, _ = trainer.get_actions(h_list, delivered, deterministic=True)
            else:
                actions, _ = trainer.get_actions(h_list, {i: [] for i in range(env.n_agents)}, deterministic=True)
                step_cost = 0.0
                
            next_obs, next_state, reward, done, _ = env.step(actions)
            ep_return += reward
            obs = next_obs
            state = next_state
            if done:
                break
                
        returns.append(ep_return)
        if channel:
            c_disc, c_bar = channel.get_episode_metrics()
            c_disc_list.append(c_disc)
            c_bar_list.append(c_bar)
        else:
            c_disc_list.append(0.0)
            c_bar_list.append(0.0)
            
    mean_ret = float(np.mean(returns))
    return {
        'mean_return': mean_ret,
        'std_return': float(np.std(returns)),
        'mean_c_disc': float(np.mean(c_disc_list)),
        'mean_c_bar': float(np.mean(c_bar_list)),
        'norm_score': env.normalize_score(mean_ret),
        'returns': returns
    }

def train_seed(mode, fixed_bw, seed, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=-10.72, R_MAX=8.70):
    torch.set_num_threads(1)
    set_seed(seed)
    env = ContinuousTrackingEnv(n_agents=3, max_cycles=25, full_obs=full_obs)
    env.set_endpoints(R_MIN, R_MAX)
    channel = BroadcastChannel(n_agents=3, message_dim=16, p_drop=0.0, delay_tau=0) if mode != 'no_comm' and not full_obs else None
    
    trainer = MAPPOTrainer(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=3,
        action_dim=env.action_dim,
        mode=mode,
        fixed_bit_width=fixed_bw,
        actor_lr=5e-4,
        critic_lr=1e-3,
        ppo_epochs=4,
        entropy_coef=0.01,
        device='cpu'
    )
    
    curve = []
    
    for update in range(n_updates):
        batch_obs, batch_states, batch_actions = [], [], []
        batch_action_lps, batch_gates, batch_gate_lps = [], [], []
        batch_bit_widths, batch_width_lps, batch_deliv_masks = [], [], []
        batch_rewards, batch_step_costs, batch_dones = [], [], []
        
        ep_returns = []
        for ep in range(eps_per_update):
            obs, state = env.reset(seed=seed * 1000 + update * 12 + ep)
            if channel:
                channel.reset()
            ep_r = 0.0
            
            for t in range(env.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer.act(obs, deterministic=False)
                
                if mode != 'no_comm' and channel is not None:
                    g_train = [1, 0, 0]
                    b_train = [fixed_bw, 0, 0]
                    quantised_msgs = [
                        trainer.quantiser(raw_msgs[i:i+1], bit_width=b_train[i]).squeeze(0) if g_train[i] == 1 else torch.zeros(16)
                        for i in range(env.n_agents)
                    ]
                    delivered, step_cost = channel.transmit_and_deliver(quantised_msgs, g_train, b_train)
                    actions, action_lps = trainer.get_actions(h_list, delivered, deterministic=False)
                    
                    mask_matrix = [[0 for _ in range(env.n_agents)] for _ in range(env.n_agents)]
                    for r_idx, items in delivered.items():
                        for it in items:
                            mask_matrix[it['sender']][r_idx] = 1
                else:
                    actions, action_lps = trainer.get_actions(h_list, {i: [] for i in range(env.n_agents)}, deterministic=False)
                    mask_matrix = [[0 for _ in range(env.n_agents)] for _ in range(env.n_agents)]
                    g_train = [0, 0, 0]
                    b_train = [0, 0, 0]
                    step_cost = 0.0
                    
                next_obs, next_state, reward, done, _ = env.step(actions)
                ep_r += reward
                
                batch_obs.append([obs[f"agent_{i}"] for i in range(env.n_agents)])
                batch_states.append(state)
                batch_actions.append([actions[f"agent_{i}"] for i in range(env.n_agents)])
                batch_action_lps.append([action_lps[f"agent_{i}"] for i in range(env.n_agents)])
                batch_gates.append(g_train)
                batch_gate_lps.append([0.0] * env.n_agents)
                batch_bit_widths.append(b_train)
                batch_width_lps.append([0.0] * env.n_agents)
                batch_deliv_masks.append(mask_matrix)
                batch_rewards.append(reward)
                batch_step_costs.append(step_cost)
                batch_dones.append(1.0 if t == env.max_cycles - 1 else 0.0)
                
                obs = next_obs
                state = next_state
                if done:
                    break
            ep_returns.append(ep_r)
            
        buf = {
            'obs': batch_obs, 'state': batch_states, 'actions': batch_actions,
            'action_lps': batch_action_lps, 'gates': batch_gates, 'gate_lps': batch_gate_lps,
            'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
            'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs,
            'dones': batch_dones
        }
        trainer.train_step(buf)
        curve.append(float(np.mean(ep_returns)))
        
    eval_metrics = evaluate_policy_on_test_suite(env, channel, trainer, test_seeds, mode=mode, fixed_bw=fixed_bw)
    eval_metrics['learning_curve'] = curve
    return eval_metrics

def run_seeds_parallel(mode, fixed_bw, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=-10.72, R_MAX=8.70):
    results = {}
    with ProcessPoolExecutor(max_workers=min(len(seeds), 5)) as executor:
        future_to_seed = {
            executor.submit(train_seed, mode, fixed_bw, s, test_seeds, full_obs, n_updates, eps_per_update, R_MIN, R_MAX): s
            for s in seeds
        }
        for fut in future_to_seed:
            s = future_to_seed[fut]
            res = fut.result()
            results[s] = res
            print(f"    Seed {s} Done. Return: {res['mean_return']:.2f}, Score: {res['norm_score']:.3f}")
    return results

def main():
    seeds = [42, 43, 44, 45, 46]
    test_seeds = [1000 + i * 17 for i in range(20)]
    os.makedirs("results/phase1", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    print("=" * 75)
    print("CONTINUOUS TIME-VARYING TRACKING POMDP EXPERIMENT")
    print(f"Training Seeds: {seeds} (N = {len(seeds)})")
    print(f"Held-out Test Suite: 20 fixed episodes ({test_seeds[0]}..{test_seeds[-1]})")
    print("=" * 75)
    
    env = ContinuousTrackingEnv(n_agents=3, max_cycles=25)
    
    # Evaluate Scripted Baselines
    print("\n[Step 1/5] Evaluating Scripted Oracle, No-Info Heuristic, & Random...")
    # Random
    r_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            acts = {f"agent_{i}": np.random.randint(0, 5) for i in range(3)}
            _, _, r, done, _ = env.step(acts)
            ep_r += r
            if done: break
        r_rets.append(ep_r)
    mean_random = float(np.mean(r_rets))
    std_random = float(np.std(r_rets))

    # Scripted Heuristic
    h_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            raw = env.env.unwrapped
            acts = {'agent_0': 0}
            for i in [1, 2]:
                pos = raw.world.agents[i].state.p_pos
                acts[f"agent_{i}"] = get_proportional_action(pos, np.array([0.0, 0.0]))
            _, _, r, done, _ = env.step(acts)
            ep_r += r
            if done: break
        h_rets.append(ep_r)
    mean_heuristic = float(np.mean(h_rets))
    std_heuristic = float(np.std(h_rets))

    # Scripted Oracle
    o_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            raw = env.env.unwrapped
            acts = {'agent_0': 0}
            for i in [1, 2]:
                pos = raw.world.agents[i].state.p_pos
                acts[f"agent_{i}"] = get_proportional_action(pos, env.target_pos)
            _, _, r, done, _ = env.step(acts)
            ep_r += r
            if done: break
        o_rets.append(ep_r)
    mean_oracle = float(np.mean(o_rets))
    std_oracle = float(np.std(o_rets))

    # Freeze Normalisation Endpoints ONCE using Scripted Baselines
    R_MIN = mean_heuristic # -10.72
    R_MAX = mean_oracle    # 8.70
    env.set_endpoints(R_MIN, R_MAX)
    
    print(f"  Random Policy Lower Bound       : Return = {mean_random:.2f} +- {std_random:.2f} (Score: {env.normalize_score(mean_random):.3f})")
    print(f"  Scripted No-Info Heuristic (R_MIN): Return = {mean_heuristic:.2f} +- {std_heuristic:.2f} (Score: {env.normalize_score(mean_heuristic):.3f})")
    print(f"  Scripted Oracle (R_MAX)          : Return = {mean_oracle:.2f} +- {std_oracle:.2f} (Score: {env.normalize_score(mean_oracle):.3f})")
    print(f"  Frozen Endpoints: R_MIN = {R_MIN:.2f}, R_MAX = {R_MAX:.2f}")

    results = defaultdict(dict)
    results['random'] = {'mean_return': mean_random, 'std_return': std_random, 'norm_score': env.normalize_score(mean_random)}
    results['heuristic'] = {'mean_return': mean_heuristic, 'std_return': std_heuristic, 'norm_score': env.normalize_score(mean_heuristic)}
    results['oracle'] = {'mean_return': mean_oracle, 'std_return': std_oracle, 'norm_score': env.normalize_score(mean_oracle)}

    # Step 2: Full-Observation Upper Bound
    print("\n[Step 2/5] Training Full-Observation Upper Bound (5 seeds in parallel)...")
    t0 = time.time()
    fo_results = run_seeds_parallel('no_comm', 0, seeds, test_seeds, full_obs=True, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
    print(f"  Full-Obs completed in {time.time()-t0:.1f}s.")
    mean_fo = float(np.mean([fo_results[s]['mean_return'] for s in seeds]))
    std_fo = float(np.std([fo_results[s]['mean_return'] for s in seeds]))
    print(f"  Full-Obs Mean Return: {mean_fo:.2f} +- {std_fo:.2f} (Score: {env.normalize_score(mean_fo):.3f})")
    results['full_obs'] = fo_results

    # Step 3: No-Communication Baseline
    print("\n[Step 3/5] Training No-Communication Baseline (5 seeds in parallel)...")
    t0 = time.time()
    results['no_comm'] = run_seeds_parallel('no_comm', 0, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
    print(f"  No-Comm completed in {time.time()-t0:.1f}s.")
    
    # Step 4: Full-Communication (b=8) Screening Baseline
    print("\n[Step 4/5] Training Full-Communication (b=8) Baseline (5 seeds in parallel)...")
    t0 = time.time()
    results['full_comm_b8'] = run_seeds_parallel('full_comm_fixed', 8, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
    print(f"  Full-Comm (b=8) completed in {time.time()-t0:.1f}s.")

    # Check 1: Paired-seed confidence interval
    print("\n" + "=" * 75)
    print("CHECK 1 EVALUATION: Full-Comm (b=8) vs No-Comm")
    print("=" * 75)
    no_scores = [results['no_comm'][s]['norm_score'] for s in seeds]
    fc_scores = [results['full_comm_b8'][s]['norm_score'] for s in seeds]
    diffs = [fc_scores[i] - no_scores[i] for i in range(len(seeds))]
    
    mean_diff, margin, ci_low, ci_high = compute_paired_ci_95(diffs)
    print(f"No-Comm Scores (5 seeds)       : {[round(x, 4) for x in no_scores]} (Mean: {np.mean(no_scores):.4f})")
    print(f"Full-Comm (b=8) Scores (5 seeds): {[round(x, 4) for x in fc_scores]} (Mean: {np.mean(fc_scores):.4f})")
    print(f"Paired Differences (Delta_s)    : {[round(x, 4) for x in diffs]}")
    print(f"Mean Difference (Delta_bar)     : {mean_diff:.4f}")
    print(f"Paired 95% Confidence Interval  : [{ci_low:.4f}, {ci_high:.4f}]")
    check1_passed = (mean_diff >= 0.10) and (ci_low > 0.0)
    print(f"Check 1 Status                  : {'PASSED' if check1_passed else 'FAILED'}")

    # Step 5: Precision Screen across fixed widths W = {1, 2, 4}
    print("\n[Step 5/5] Precision Screening across fixed widths W = {1, 2, 4} (5 seeds each in parallel)...")
    for bw in [1, 2, 4]:
        mode_key = f"full_comm_b{bw}"
        print(f"  Training {mode_key}...")
        t0 = time.time()
        results[mode_key] = run_seeds_parallel('full_comm_fixed', bw, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
        print(f"    {mode_key} completed in {time.time()-t0:.1f}s.")

    # Check 2: Precision Screening Flatness & Significance Test
    print("\n" + "=" * 75)
    print("CHECK 2 EVALUATION & STATISTICAL SIGNIFICANCE TEST ACROSS W = {1, 2, 4, 8}")
    print("=" * 75)
    width_scores = {}
    per_seed_width = {bw: [] for bw in [1, 2, 4, 8]}
    for bw in [1, 2, 4, 8]:
        mode_key = f"full_comm_b{bw}"
        scores = [results[mode_key][s]['norm_score'] for s in seeds]
        per_seed_width[bw] = scores
        width_scores[bw] = float(np.mean(scores))
        print(f"  Width b={bw}: Mean Normalised Score = {width_scores[bw]:.4f} (Seeds: {[round(x, 3) for x in scores]})")
        
    pairwise_diffs = []
    widths = [1, 2, 4, 8]
    for i in range(len(widths)):
        for j in range(i + 1, len(widths)):
            d = abs(width_scores[widths[i]] - width_scores[widths[j]])
            pairwise_diffs.append(d)
            print(f"  |Score(b={widths[i]}) - Score(b={widths[j]})| = {d:.4f}")
            
    max_pairwise_diff = max(pairwise_diffs)
    check2_flat = max_pairwise_diff < 0.10
    print(f"Max Pairwise Difference: {max_pairwise_diff:.4f} (Flatness Threshold < 0.10)")
    print(f"Flatness Check Outcome: {'RETURNS ARE FLAT (< 0.10) -> Learned Precision Deprioritised' if check2_flat else 'RETURNS NOT FLAT (>= 0.10)'}")

    # Significance Test on Precision Gap: Paired Student-t test between b=8 and b=1
    diffs_8_vs_1 = [per_seed_width[8][i] - per_seed_width[1][i] for i in range(len(seeds))]
    t_stat, p_val = stats.ttest_rel(per_seed_width[8], per_seed_width[1])
    d_mean, _, d_ci_low, d_ci_high = compute_paired_ci_95(diffs_8_vs_1)
    
    # Monotonicity test: Linear regression across all (b, score) pairs
    all_b = []
    all_s = []
    for bw in [1, 2, 4, 8]:
        for score_val in per_seed_width[bw]:
            all_b.append(bw)
            all_s.append(score_val)
    slope, intercept, r_val, p_val_slope, std_err = stats.linregress(all_b, all_s)
    
    print("\nStatistical Significance Tests:")
    print(f"  1. Paired t-test (b=8 vs b=1):")
    print(f"     Mean Difference: {d_mean:.4f}, 95% CI: [{d_ci_low:.4f}, {d_ci_high:.4f}]")
    print(f"     t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    print(f"     Significant at alpha=0.05: {'YES' if p_val < 0.05 else 'NO (Null result)'}")
    print(f"  2. Linear Trend across Precision Levels W:")
    print(f"     Slope = {slope:.5f}/bit, r = {r_val:.4f}, p-value = {p_val_slope:.4f}")
    print(f"     Monotonic Trend Significant: {'YES' if p_val_slope < 0.05 else 'NO (Null result)'}")

    # Save JSONL raw records
    jsonl_records = []
    for mode_name in ['random', 'heuristic', 'oracle']:
        jsonl_records.append({
            'mode': mode_name, 'seed': 'all',
            'mean_return': results[mode_name]['mean_return'],
            'std_return': results[mode_name]['std_return'],
            'mean_c_disc': 0.0, 'mean_c_bar': 0.0,
            'norm_score': results[mode_name]['norm_score']
        })
    for mode_name in ['full_obs', 'no_comm', 'full_comm_b8', 'full_comm_b1', 'full_comm_b2', 'full_comm_b4']:
        for s, metrics in results[mode_name].items():
            jsonl_records.append({
                'mode': mode_name, 'seed': s,
                'mean_return': metrics['mean_return'],
                'std_return': metrics['std_return'],
                'mean_c_disc': metrics['mean_c_disc'],
                'mean_c_bar': metrics['mean_c_bar'],
                'norm_score': metrics['norm_score']
            })
    save_jsonl(jsonl_records, "results/phase1/continuous_results.jsonl")

    # Save YAML summary
    summary = {
        'endpoints': {'R_MIN': float(R_MIN), 'R_MAX': float(R_MAX)},
        'baselines': {
            'random': {'mean': mean_random, 'std': std_random, 'norm_score': env.normalize_score(mean_random)},
            'heuristic': {'mean': mean_heuristic, 'std': std_heuristic, 'norm_score': env.normalize_score(mean_heuristic)},
            'oracle': {'mean': mean_oracle, 'std': std_oracle, 'norm_score': env.normalize_score(mean_oracle)}
        },
        'check1': {
            'no_comm_scores': [float(x) for x in no_scores],
            'full_comm_b8_scores': [float(x) for x in fc_scores],
            'paired_differences': [float(x) for x in diffs],
            'mean_difference': float(mean_diff),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
            'passed': bool(check1_passed)
        },
        'check2': {
            'width_scores': {str(k): float(v) for k, v in width_scores.items()},
            'max_pairwise_diff': float(max_pairwise_diff),
            'flat': bool(check2_flat),
            'learned_precision_deprioritised': bool(check2_flat),
            'significance_8_vs_1': {
                't_stat': float(t_stat), 'p_value': float(p_val),
                'mean_diff': float(d_mean), 'ci_low': float(d_ci_low), 'ci_high': float(d_ci_high),
                'significant_05': bool(p_val < 0.05)
            },
            'linear_trend': {
                'slope': float(slope), 'r_val': float(r_val), 'p_val': float(p_val_slope),
                'significant_05': bool(p_val_slope < 0.05)
            }
        }
    }
    with open("results/phase1/continuous_summary.yaml", "w") as f:
        yaml.dump(summary, f)

    # Plot comprehensive learning curves
    plt.figure(figsize=(11, 6))
    for mode_name in ['no_comm', 'full_comm_b8', 'full_comm_b1', 'full_comm_b2', 'full_comm_b4', 'full_obs']:
        curves = [results[mode_name][s]['learning_curve'] for s in seeds]
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        updates_axis = np.arange(1, len(mean_curve) + 1)
        plt.plot(updates_axis, mean_curve, label=f"{mode_name} (final: {mean_curve[-1]:.1f})")
        plt.fill_between(updates_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)
        
    plt.axhline(mean_oracle, color='darkgreen', linestyle='-', linewidth=1.5, label=f'Scripted Oracle ({mean_oracle:.1f})')
    plt.axhline(mean_heuristic, color='purple', linestyle='--', linewidth=1.5, label=f'Scripted No-Info Heuristic ({mean_heuristic:.1f})')
    plt.axhline(mean_random, color='gray', linestyle=':', linewidth=1.5, label=f'Random Policy ({mean_random:.1f})')
    
    plt.xlabel("PPO Update (12 eps / update)")
    plt.ylabel("Episode Return")
    plt.title("Continuous Time-Varying POMDP: Learning Curves & Baselines (5 Seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/figures/continuous_learning_curves.png", dpi=300)
    plt.close()
    
    print("\nSaved learning curves to: results/figures/continuous_learning_curves.png")
    print("Saved raw records to: results/phase1/continuous_results.jsonl")
    print("Saved summary report to: results/phase1/continuous_summary.yaml")
    print("=" * 75)

if __name__ == '__main__':
    main()
