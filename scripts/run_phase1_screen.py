import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.po_mpe import PartiallyObservableSpreadEnv
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from src.utils.metrics import compute_paired_ci_95, save_jsonl

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy_on_test_suite(env, channel, trainer, test_seeds, mode='full_comm', fixed_bw=8):
    """
    Evaluates policy deterministically on a fixed held-out test suite.
    Uses greedy discrete choices (deterministic=True) as mandated.
    """
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
                
            next_obs, next_state, reward, done, _ = env.step(actions)
            ep_return += reward
            obs = next_obs
            state = next_state
            if done:
                break
                
        if channel and mode != 'no_comm':
            c_disc, c_bar = channel.get_episode_metrics()
        else:
            c_disc, c_bar = 0.0, 0.0
            
        returns.append(ep_return)
        c_disc_list.append(c_disc)
        c_bar_list.append(c_bar)
        
    mean_ret = float(np.mean(returns))
    return {
        'mean_return': mean_ret,
        'std_return': float(np.std(returns)),
        'mean_c_disc': float(np.mean(c_disc_list)),
        'mean_c_bar': float(np.mean(c_bar_list)),
        'norm_score': env.normalize_score(mean_ret),
        'returns': returns
    }

def train_seed(mode, fixed_bw, seed, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=-25.0, R_MAX=0.0):
    torch.set_num_threads(1)
    set_seed(seed)
    env = PartiallyObservableSpreadEnv(n_agents=3, max_cycles=25, full_obs=full_obs)
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

def run_seeds_parallel(mode, fixed_bw, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=-25.0, R_MAX=0.0):
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
    print("PHASE 1 SCREENING WITH EMPIRICAL RANDOM & FULL-OBS BOUNDS")
    print(f"Training Seeds: {seeds} (N = {len(seeds)})")
    print(f"Held-out Test Suite: 20 fixed episodes ({test_seeds[0]}..{test_seeds[-1]})")
    print("=" * 75)
    
    # Step 1: Random Policy Lower Bound Evaluation
    print("\n[Step 1/5] Evaluating Random Policy Lower Bound...")
    test_env = PartiallyObservableSpreadEnv(n_agents=3, max_cycles=25)
    rand_rets = []
    for s in test_seeds:
        test_env.reset(seed=s)
        ep_r = 0.0
        for _ in range(test_env.max_cycles):
            acts = {f"agent_{i}": np.random.randint(0, 5) for i in range(3)}
            _, _, r, _, _ = test_env.step(acts)
            ep_r += r
        rand_rets.append(ep_r)
    mean_random = float(np.mean(rand_rets))
    std_random = float(np.std(rand_rets))
    print(f"  Random Policy Lower Bound: Mean Return = {mean_random:.2f} +- {std_random:.2f}")
    
    # Documented endpoints for normalised task score
    R_MIN = -25.0
    R_MAX = 0.0
    print(f"\nDocumented Endpoints for Normalised Task Score:")
    print(f"  R_MIN (Documented Blind Lower Bound): {R_MIN:.2f}")
    print(f"  R_MAX (Documented Target Ceiling)   : {R_MAX:.2f}")
    test_env.set_endpoints(R_MIN, R_MAX)
    
    # Step 2: Full-Observation Upper Bound across 5 seeds
    print("\n[Step 2/5] Training Full-Observation Upper Bound (5 seeds in parallel)...")
    t0 = time.time()
    fo_results = run_seeds_parallel('no_comm', 0, seeds, test_seeds, full_obs=True, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
    print(f"  Full-Obs completed in {time.time()-t0:.1f}s.")
    mean_full_obs = float(np.mean([fo_results[s]['mean_return'] for s in seeds]))
    std_full_obs = float(np.std([fo_results[s]['mean_return'] for s in seeds]))
    print(f"  Full-Observation Upper Bound: Mean Return = {mean_full_obs:.2f} +- {std_full_obs:.2f}")
    
    results = defaultdict(dict)
    results['random'] = {
        'mean_return': mean_random,
        'std_return': std_random,
        'norm_score': test_env.normalize_score(mean_random)
    }
    results['full_obs'] = fo_results
    
    # Save config
    config = {
        'phase': 1,
        'seeds': seeds,
        'test_seeds': test_seeds,
        'n_updates': 25,
        'eps_per_update': 12,
        'endpoints': {'R_MIN': float(R_MIN), 'R_MAX': float(R_MAX)},
        'allowed_widths': [1, 2, 4, 8]
    }
    with open("configs/phase1_screen.yaml", "w") as f:
        yaml.dump(config, f)

    # Step 3: No-Communication Baseline
    print("\n[Step 3/5] Training No-Communication Baseline (5 seeds in parallel)...")
    t0 = time.time()
    results['no_comm'] = run_seeds_parallel('no_comm', 0, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
    print(f"  No-Comm completed in {time.time()-t0:.1f}s.")
        
    # Step 4: Full-Communication (b=8) Screening Baseline
    print("\n[Step 4/5] Training Full-Communication (b=8) Screening Baseline (5 seeds in parallel)...")
    t0 = time.time()
    results['full_comm_b8'] = run_seeds_parallel('full_comm_fixed', 8, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
    print(f"  Full-Comm (b=8) completed in {time.time()-t0:.1f}s.")

    # Check 1: Paired-seed confidence interval on normalised scores
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
    print(f"Threshold Requirement           : Mean Diff >= 0.10 AND CI lower bound > 0.0")
    print(f"Check 1 Status                  : {'PASSED' if check1_passed else 'FAILED'}")
    
    # Step 5: Precision Screen across fixed bit-widths W = {1, 2, 4}
    print("\n[Step 5/5] Precision Screening across fixed widths W = {1, 2, 4} (5 seeds each in parallel)...")
    for bw in [1, 2, 4]:
        mode_key = f"full_comm_b{bw}"
        print(f"  Training {mode_key}...")
        t0 = time.time()
        results[mode_key] = run_seeds_parallel('full_comm_fixed', bw, seeds, test_seeds, full_obs=False, n_updates=25, eps_per_update=12, R_MIN=R_MIN, R_MAX=R_MAX)
        print(f"    {mode_key} completed in {time.time()-t0:.1f}s.")
            
    # Check 2: Precision Screening Flatness
    print("\n" + "=" * 75)
    print("CHECK 2 EVALUATION: Precision Screening Flatness Across W = {1, 2, 4, 8}")
    print("=" * 75)
    width_scores = {}
    for bw in [1, 2, 4, 8]:
        mode_key = f"full_comm_b{bw}"
        scores = [results[mode_key][s]['norm_score'] for s in seeds]
        width_scores[bw] = float(np.mean(scores))
        print(f"  Width b={bw}: Mean Normalised Score = {width_scores[bw]:.4f}")
        
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
    print(f"Check 2 Outcome: {'RETURNS ARE FLAT (< 0.10) -> Learned Precision Deprioritised' if check2_flat else 'RETURNS NOT FLAT (>= 0.10)'}")

    # Save JSONL raw logs
    jsonl_records = []
    for mode_name, seed_dict in results.items():
        if mode_name == 'random':
            jsonl_records.append({
                'mode': 'random',
                'seed': 'all',
                'mean_return': seed_dict['mean_return'],
                'std_return': seed_dict['std_return'],
                'mean_c_disc': 0.0,
                'mean_c_bar': 0.0,
                'norm_score': seed_dict['norm_score']
            })
        else:
            for s, metrics in seed_dict.items():
                jsonl_records.append({
                    'mode': mode_name,
                    'seed': s,
                    'mean_return': metrics['mean_return'],
                    'std_return': metrics['std_return'],
                    'mean_c_disc': metrics['mean_c_disc'],
                    'mean_c_bar': metrics['mean_c_bar'],
                    'norm_score': metrics['norm_score']
                })
    save_jsonl(jsonl_records, "results/phase1/phase1_results.jsonl")
    
    # Save YAML summary
    summary = {
        'endpoints': {'R_MIN': float(R_MIN), 'R_MAX': float(R_MAX)},
        'random_lower_bound': {'mean': mean_random, 'std': std_random, 'norm_score': float(test_env.normalize_score(mean_random))},
        'full_obs_upper_bound': {'mean': mean_full_obs, 'std': std_full_obs, 'norm_score': float(test_env.normalize_score(mean_full_obs))},
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
            'learned_precision_deprioritised': bool(check2_flat)
        }
    }
    with open("results/phase1/phase1_summary.yaml", "w") as f:
        yaml.dump(summary, f)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    for mode_name in ['no_comm', 'full_comm_b8', 'full_comm_b1', 'full_comm_b2', 'full_comm_b4']:
        curves = [results[mode_name][s]['learning_curve'] for s in seeds]
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        updates_axis = np.arange(1, len(mean_curve) + 1)
        plt.plot(updates_axis, mean_curve, label=f"{mode_name} (mean: {mean_curve[-1]:.1f})")
        plt.fill_between(updates_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        
    plt.axhline(mean_random, color='gray', linestyle='--', label=f'Random Lower Bound ({mean_random:.1f})')
    plt.axhline(mean_full_obs, color='black', linestyle=':', label=f'Full-Obs Upper Bound ({mean_full_obs:.1f})')
    plt.xlabel("PPO Update")
    plt.ylabel("Episode Return")
    plt.title("Phase 1 Screening: Learning Curves across 5 Seeds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/phase1_learning_curves.png", dpi=300)
    plt.close()
    
    print("\nSaved learning curves to: results/figures/phase1_learning_curves.png")
    print("Saved raw records to: results/phase1/phase1_results.jsonl")
    print("Saved summary report to: results/phase1/phase1_summary.yaml")
    print("=" * 75)

if __name__ == '__main__':
    main()
