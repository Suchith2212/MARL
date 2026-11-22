"""
Parallel dual step size and PID Lagrangian tuning on validation seeds [42, 43].
Tests combination of (K_p, alpha_I) on b=4, B=16.
Measures signed cost error (bar_C - B) / B.
"""
import torch, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.primal_dual_zoh import ZOHMAPPOTrainer

VAL_SEEDS = [42, 43]
BUDGET = 16
BW = 4
TOTAL_UPDATES = 125 # 200k steps

configs = [
    {'name': 'I_only', 'K_p': 0.0, 'alpha_I': 0.01},
    {'name': 'PID_low', 'K_p': 0.005, 'alpha_I': 0.005},
    {'name': 'PID_mid', 'K_p': 0.01, 'alpha_I': 0.01},
    {'name': 'PID_high', 'K_p': 0.02, 'alpha_I': 0.02},
]

def run_worker(args):
    cfg, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = BatchedTrackingEnv(num_envs=64, max_cycles=25, stage='dynamic', full_obs=False)
    trainer = ZOHMAPPOTrainer(
        mode='primal_dual', fixed_bw=BW, budget=BUDGET, d=16, T=25,
        K_p=cfg['K_p'], alpha_I=cfg['alpha_I'], warmup_updates=25
    )
    for upd in range(1, TOTAL_UPDATES + 1):
        obs, state = env.reset(seed=seed * 1000 + upd)
        last_msg = torch.zeros(64, 16)
        tau = torch.zeros(64, 1)
        bufs = {k: [] for k in ['states','scout_obs','rover1_obs','rover2_obs','held_msg','tau',
                                  'gates','gate_lps','p1_hat','p2_hat','lp1','lp2','raw_rewards','step_costs','dones']}
        for t in range(25):
            bufs['held_msg'].append(last_msg); bufs['tau'].append(tau)
            t_est, g, glp, (p1, p2), (lp1, lp2), sc, nl_msg, n_tau = trainer.act(obs, last_msg, tau, t)
            no, ns, r, d, _ = env.step_with_target_estimates(t_est)
            bufs['states'].append(state); bufs['scout_obs'].append(obs['agent_0'])
            bufs['rover1_obs'].append(obs['agent_1'][:, :10]); bufs['rover2_obs'].append(obs['agent_2'][:, :10])
            bufs['gates'].append(g); bufs['gate_lps'].append(glp)
            bufs['p1_hat'].append(p1); bufs['p2_hat'].append(p2)
            bufs['lp1'].append(lp1); bufs['lp2'].append(lp2)
            bufs['raw_rewards'].append(r); bufs['step_costs'].append(sc); bufs['dones'].append(d.float())
            obs, state = no, ns; last_msg, tau = nl_msg, n_tau
        trainer.train_step({k: torch.stack(v) for k, v in bufs.items()})

    # Eval
    eval_env = BatchedTrackingEnv(500, 25, stage='dynamic', full_obs=False)
    e_obs, _ = eval_env.reset(seed=99999)
    e_msg, e_tau = torch.zeros(500, 16), torch.zeros(500, 1)
    tot_r, tot_c = torch.zeros(500), torch.zeros(500)
    with torch.no_grad():
        for t in range(25):
            t_est, g, _, _, _, sc, nl_msg, n_tau = trainer.act(e_obs, e_msg, e_tau, t, deterministic=True)
            e_obs, _, r, _, _ = eval_env.step_with_target_estimates(t_est)
            tot_r += r; tot_c += sc; e_msg, e_tau = nl_msg, n_tau
            
    c = float(tot_c.mean() / 25.0)
    ret = float(tot_r.mean())
    return cfg['name'], seed, c, ret, float(trainer.dual_lambda)

def main():
    print(f"=== TUNING PID LAGRANGIAN (b={BW}, B={BUDGET}) ON VALIDATION SEEDS {VAL_SEEDS} ===")
    work = [(cfg, s) for cfg in configs for s in VAL_SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_worker, work))
    print(f"Tuning finished in {time.time()-t0:.1f}s")
    
    for cfg in configs:
        res_cfg = [r for r in results if r[0] == cfg['name']]
        costs = [r[2] for r in res_cfg]
        rets = [r[3] for r in res_cfg]
        lams = [r[4] for r in res_cfg]
        mean_c = np.mean(costs)
        mean_r = np.mean(rets)
        err = (mean_c - BUDGET) / BUDGET
        print(f"[{cfg['name']:<8}] Cost: {mean_c:5.2f} (Err: {err:+.3f}), Return: {mean_r:+.2f}, Final lambda: {np.mean(lams):.4f}")

if __name__ == '__main__':
    main()
