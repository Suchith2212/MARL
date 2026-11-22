import numpy as np
from src.envs.po_mpe import PartiallyObservableSpreadEnv

def get_proportional_action(agent_pos, target_pos):
    diff = target_pos - agent_pos
    dx, dy = diff[0], diff[1]
    if np.linalg.norm(diff) < 0.05:
        return 0
    if abs(dx) > abs(dy):
        return 1 if dx < 0 else 2 # left/right
    else:
        return 3 if dy < 0 else 4 # down/up

def main():
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = PartiallyObservableSpreadEnv(n_agents=3, max_cycles=25)
    
    # 1. Scripted Oracle: Knows target_idx, navigates to target landmark
    oracle_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            raw = env.env.unwrapped
            target = env.lm_pos[env.target_idx]
            acts = {'agent_0': 0}
            for i in [1, 2]:
                pos = raw.world.agents[i].state.p_pos
                acts[f"agent_{i}"] = get_proportional_action(pos, target)
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        oracle_rets.append(ep_r)
        
    # 2. Scripted No-Information Heuristic: Moves to Bayes centroid (0.0, 0.0)
    heur_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            raw = env.env.unwrapped
            centroid = np.array([0.0, 0.0]) # midpoint of L0 and L1
            acts = {'agent_0': 0}
            for i in [1, 2]:
                pos = raw.world.agents[i].state.p_pos
                acts[f"agent_{i}"] = get_proportional_action(pos, centroid)
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        heur_rets.append(ep_r)

    # 3. Random Policy
    rand_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            acts = {f"agent_{i}": np.random.randint(0, 5) for i in range(3)}
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        rand_rets.append(ep_r)

    print(f"Random Policy            : Mean = {np.mean(rand_rets):.2f} +- {np.std(rand_rets):.2f}")
    print(f"Scripted No-Info Heuristic: Mean = {np.mean(heur_rets):.2f} +- {np.std(heur_rets):.2f}")
    print(f"Scripted Oracle          : Mean = {np.mean(oracle_rets):.2f} +- {np.std(oracle_rets):.2f}")

if __name__ == '__main__':
    main()
