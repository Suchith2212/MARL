import os
import sys
import yaml
import numpy as np
import torch
from collections import defaultdict
import time

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

def evaluate_primal_dual_policy(env, channel, trainer, test_seeds, budget_B):
    """
    Evaluates policy deterministically on held-out test episodes.
    Calculates mean return, realized C (discounted-normalised), plain C_bar,
    and empirical episode violation rate nu(B) = Pr(C_e > B).
    """
    returns = []
    c_disc_list = []
    c_bar_list = []
    violations = []
    
    for s in test_seeds:
        obs, state = env.reset(seed=s)
        channel.reset()
        ep_return = 0.0
        
        for t in range(env.max_cycles):
            # Deterministic greedy choices at evaluation
            h_list, raw_msgs, gates, _, bit_widths, _ = trainer.act(obs, deterministic=True)
            
            quantised_msgs = []
            for i in range(env.n_agents):
                if gates[i] == 1 and bit_widths[i] > 0:
                    qm = trainer.quantiser(raw_msgs[i:i+1], bit_width=bit_widths[i]).squeeze(0)
                else:
                    qm = torch.zeros_like(raw_msgs[i])
                quantised_msgs.append(qm)
                
            delivered, step_cost = channel.transmit_and_deliver(quantised_msgs, gates, bit_widths)
            actions, _ = trainer.get_actions(h_list, delivered, deterministic=True)
            
            next_obs, next_state, reward, done, _ = env.step(actions)
            ep_return += reward
            obs = next_obs
            state = next_state
            if done:
                break
                
        c_disc, c_bar = channel.get_episode_metrics()
        returns.append(ep_return)
        c_disc_list.append(c_disc)
        c_bar_list.append(c_bar)
        violations.append(1.0 if c_disc > budget_B else 0.0)
        
    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'mean_c_disc': float(np.mean(c_disc_list)),
        'mean_c_bar': float(np.mean(c_bar_list)),
        'violation_rate_nu': float(np.mean(violations)),
        'final_lambda': float(trainer.lambda_cost),
        'norm_score': env.normalize_score(float(np.mean(returns)))
    }

def train_primal_dual(budget_B, seed, test_seeds, fixed_bw=4, n_updates=40, eps_per_update=6):
    set_seed(seed)
    env = PartiallyObservableSpreadEnv(n_agents=3, r_obs=0.50, max_cycles=25)
    channel = BroadcastChannel(n_agents=3, message_dim=16, p_drop=0.0, delay_tau=0)
    
    trainer = MAPPOTrainer(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=3,
        action_dim=env.action_dim,
        mode='learned_gate_fixed',
        fixed_bit_width=fixed_bw,
        budget_B=budget_B,
        dual_lr=0.0001,
        actor_lr=4e-4,
        critic_lr=1e-3,
        ppo_epochs=4,
        entropy_coef=0.01,
        device='cpu'
    )
    
    for update in range(n_updates):
        batch_obs, batch_states, batch_actions = [], [], []
        batch_action_lps, batch_gates, batch_gate_lps = [], [], []
        batch_bit_widths, batch_width_lps, batch_deliv_masks = [], [], []
        batch_rewards, batch_step_costs, batch_dones = [], [], []
        
        for ep in range(eps_per_update):
            obs, state = env.reset(seed=seed * 1000 + update * 10 + ep)
            channel.reset()
            
            for t in range(env.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer.act(obs, deterministic=False)
                
                quantised_msgs = []
                for i in range(env.n_agents):
                    if gates[i] == 1 and bit_widths[i] > 0:
                        qm = trainer.quantiser(raw_msgs[i:i+1], bit_width=bit_widths[i]).squeeze(0)
                    else:
                        qm = torch.zeros_like(raw_msgs[i])
                    quantised_msgs.append(qm)
                    
                delivered, step_cost = channel.transmit_and_deliver(quantised_msgs, gates, bit_widths)
                actions, action_lps = trainer.get_actions(h_list, delivered, deterministic=False)
                
                mask_matrix = [[0 for _ in range(env.n_agents)] for _ in range(env.n_agents)]
                for r_idx, items in delivered.items():
                    for it in items:
                        mask_matrix[it['sender']][r_idx] = 1
                        
                next_obs, next_state, reward, done, _ = env.step(actions)
                
                batch_obs.append([obs[f"agent_{i}"] for i in range(env.n_agents)])
                batch_states.append(state)
                batch_actions.append([actions[f"agent_{i}"] for i in range(env.n_agents)])
                batch_action_lps.append([action_lps[f"agent_{i}"] for i in range(env.n_agents)])
                batch_gates.append(gates)
                batch_gate_lps.append(gate_lps)
                batch_bit_widths.append(bit_widths)
                batch_width_lps.append(width_lps)
                batch_deliv_masks.append(mask_matrix)
                batch_rewards.append(reward)
                batch_step_costs.append(step_cost)
                batch_dones.append(1.0 if t == env.max_cycles - 1 else 0.0)
                
                obs = next_obs
                state = next_state
                if done:
                    break
                    
        buf = {
            'obs': batch_obs, 'state': batch_states, 'actions': batch_actions,
            'action_lps': batch_action_lps, 'gates': batch_gates, 'gate_lps': batch_gate_lps,
            'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
            'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs,
            'dones': batch_dones
        }
        trainer.train_step(buf)
        
    eval_metrics = evaluate_primal_dual_policy(env, channel, trainer, test_seeds, budget_B)
    return eval_metrics

def main():
    seeds = [42, 43, 44, 45, 46]
    test_seeds = [1000 + i * 17 for i in range(20)]
    budgets = [16.0, 32.0, 64.0, 128.0]
    
    os.makedirs("results/phase2", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    config = {
        'phase': 2,
        'seeds': seeds,
        'test_seeds': test_seeds,
        'budgets': budgets,
        'fixed_bit_width': 4,
        'dual_lr': 0.05,
        'actor_lr': 4e-4,
        'critic_lr': 1e-3,
        'n_updates': 40,
        'eps_per_update': 6
    }
    with open("configs/phase2_primal_dual.yaml", "w") as f:
        yaml.dump(config, f)
        
    print("=" * 70)
    print("STARTING PHASE 2: PRIMAL-DUAL BUDGET CONTROL SWEEP (RQ2)")
    print(f"Budgets: {budgets} payload bits/step")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    
    records = []
    summary_by_budget = defaultdict(list)
    
    for B in budgets:
        print(f"\n--- Sweeping Budget Target B = {B} bits/step ---")
        for s in seeds:
            t0 = time.time()
            print(f"  Training Seed {s} (Budget B={B})...", end=" ", flush=True)
            metrics = train_primal_dual(budget_B=B, seed=s, test_seeds=test_seeds, fixed_bw=4, n_updates=40, eps_per_update=6)
            print(f"Done in {time.time()-t0:.1f}s | Return: {metrics['mean_return']:.2f} | Realized C: {metrics['mean_c_disc']:.1f} | Plain C_bar: {metrics['mean_c_bar']:.1f} | nu(B): {metrics['violation_rate_nu']:.2f} | Lambda: {metrics['final_lambda']:.3f}")
            
            rec = {
                'method': 'primal_dual_learned_gate',
                'budget_target_B': B,
                'seed': s,
                **metrics
            }
            records.append(rec)
            summary_by_budget[B].append(metrics)
            
    # Save all raw records
    save_jsonl(records, "results/phase2/phase2_results.jsonl")
    
    # Compute summary statistics
    budget_stats = {}
    for B, m_list in summary_by_budget.items():
        mean_ret = float(np.mean([m['mean_return'] for m in m_list]))
        std_ret = float(np.std([m['mean_return'] for m in m_list]))
        mean_c_disc = float(np.mean([m['mean_c_disc'] for m in m_list]))
        mean_c_bar = float(np.mean([m['mean_c_bar'] for m in m_list]))
        mean_nu = float(np.mean([m['violation_rate_nu'] for m in m_list]))
        mean_lambda = float(np.mean([m['final_lambda'] for m in m_list]))
        
        budget_stats[B] = {
            'mean_return': mean_ret,
            'std_return': std_ret,
            'mean_realized_C': mean_c_disc,
            'mean_plain_C_bar': mean_c_bar,
            'violation_rate_nu': mean_nu,
            'mean_final_lambda': mean_lambda
        }
        
    with open("results/phase2/phase2_summary.yaml", "w") as f:
        yaml.dump(budget_stats, f)
        
    print("\n" + "=" * 70)
    print("PHASE 2 EXPERIMENT COMPLETE: SUMMARY TABLE (RQ2)")
    print(f"{'Budget B':<10} | {'Mean Return':<12} | {'Realized C':<12} | {'Plain C_bar':<12} | {'nu(B)':<8} | {'Lambda':<8}")
    print("-" * 70)
    for B, stat in budget_stats.items():
        print(f"{B:<10.1f} | {stat['mean_return']:<12.2f} | {stat['mean_realized_C']:<12.1f} | {stat['mean_plain_C_bar']:<12.1f} | {stat['violation_rate_nu']:<8.3f} | {stat['mean_final_lambda']:<8.3f}")
    print("=" * 70)
    print("Raw results saved: results/phase2/phase2_results.jsonl")
    print("Summary saved:     results/phase2/phase2_summary.yaml")

if __name__ == "__main__":
    main()
