import numpy as np

from .env import GasStorageEnv


class NumpyQNet:
    """Small two-layer Q-network trained with manual backprop.

    This keeps the example installable in minimal Python environments while
    still matching the DQN idea: approximate Q(s,a), bootstrap targets, replay.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 48, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.25, size=(obs_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, 0.20, size=(hidden, n_actions))
        self.b2 = np.zeros(n_actions)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x2 = np.atleast_2d(x).astype(float)
        h = np.maximum(0.0, x2 @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def copy(self):
        other = NumpyQNet(self.w1.shape[0], self.w2.shape[1], self.w1.shape[1])
        other.w1 = self.w1.copy()
        other.b1 = self.b1.copy()
        other.w2 = self.w2.copy()
        other.b2 = self.b2.copy()
        return other

    def train_batch(self, states, actions, targets, lr: float) -> float:
        n = states.shape[0]
        z1 = states @ self.w1 + self.b1
        h = np.maximum(0.0, z1)
        q = h @ self.w2 + self.b2
        pred = q[np.arange(n), actions]
        err = pred - targets
        loss = float(np.mean(err * err))

        grad_q = np.zeros_like(q)
        grad_q[np.arange(n), actions] = 2.0 * err / n
        grad_w2 = h.T @ grad_q
        grad_b2 = grad_q.sum(axis=0)
        grad_h = grad_q @ self.w2.T
        grad_z1 = grad_h * (z1 > 0)
        grad_w1 = states.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        for grad in (grad_w1, grad_b1, grad_w2, grad_b2):
            np.clip(grad, -5.0, 5.0, out=grad)

        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1
        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        return loss


def train_dqn(
    episodes: int = 1200,
    seed: int = 1234,
    gamma: float = 1.0,
    batch_size: int = 128,
    lr: float = 3e-4,
) -> dict:
    rng = np.random.default_rng(seed)
    env = GasStorageEnv(seed=seed)
    q = NumpyQNet(env.obs_dim, env.n_actions, seed=seed)
    target = q.copy()
    replay = []
    max_replay = 20_000
    losses = []
    episode_rewards = []

    eps_start, eps_end = 0.9, 0.05
    for ep in range(episodes):
        obs = env.reset()
        total = 0.0
        done = False
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-ep / 350.0)
        while not done:
            feasible = env.feasible_action_indices()
            if rng.random() < epsilon:
                action = int(rng.choice(feasible))
            else:
                qvals = q.predict(obs)[0]
                masked = np.full(env.n_actions, -1e9)
                masked[feasible] = qvals[feasible]
                action = int(masked.argmax())
            next_obs, reward, done, _ = env.step(action)
            if len(replay) >= max_replay:
                replay.pop(0)
            replay.append((obs.copy(), action, reward, next_obs.copy(), done))
            obs = next_obs
            total += reward

            if len(replay) >= batch_size:
                idx = rng.choice(len(replay), size=batch_size, replace=False)
                batch = [replay[int(i)] for i in idx]
                s, a, r, ns, d = zip(*batch)
                states = np.asarray(s, dtype=float)
                actions = np.asarray(a, dtype=int)
                rewards = np.asarray(r, dtype=float)
                next_states = np.asarray(ns, dtype=float)
                dones = np.asarray(d, dtype=float)
                targets = rewards + gamma * (1.0 - dones) * target.predict(next_states).max(axis=1)
                losses.append(q.train_batch(states, actions, targets, lr=lr))

        episode_rewards.append(total)
        if ep % 50 == 0:
            target = q.copy()

    return {"q_net": q, "episode_rewards": np.asarray(episode_rewards), "losses": np.asarray(losses)}


def dqn_policy_fn(model: dict):
    q = model["q_net"]

    def policy(obs, env: GasStorageEnv) -> int:
        feasible = env.feasible_action_indices()
        qvals = q.predict(obs)[0]
        masked = np.full(env.n_actions, -1e9)
        masked[feasible] = qvals[feasible]
        return int(masked.argmax())

    return policy
