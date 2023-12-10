from collections import OrderedDict
import numpy as np
import copy
from cs285.networks.mlp_policy import MLPPolicy
import gym
import cv2
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List
import torch
import numpy as np
from tqdm import tqdm
from math import sqrt
from torch.nn import Conv2d, Linear
import torch
from scipy.linalg import svd


############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="rgb_array")
            
            if isinstance(img, list):
                img = img[0]

            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob to decide what to do
        ac = policy.get_action(ob)

        # TODO: take that action and get reward and next ob
        next_ob, rew, done, info = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done = done or steps > max_length  # HINT: this is either 0 or 1

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])

    env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])


def get_weights(model):
    flat_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            flat_weights.append(param.data.cpu().numpy().flatten())
    flat_weights = np.concatenate(flat_weights)
    return flat_weights


def get_weights_by_layer(model):
    flat_weights_by_layer = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            flat_weights_by_layer.append(param.data.cpu().numpy().flatten())
    return flat_weights_by_layer


#### Plasticity Loss Metrics
def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix
    Refer to the corresponding functions for their definitions
    :param m: (float np array) a rectangular matrix
    :param prop: (float) proportion used for computing the approximate rank
    :param use_scipy: (bool) indicates whether to compute the singular values in the cpu, only matters when using
                                  a gpu
    :return: (torch int32) rank, (torch float32) effective rank, (torch int32) approximate rank
    """
    if use_scipy:
        np_m = m.cpu().numpy()
        sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)
    else:
        sv = torch.linalg.svdvals(m)    # for large matrices, svdvals may fail to converge in gpu, but not cpu
    rank = torch.count_nonzero(sv).to(torch.int32)
    effective_rank = compute_effective_rank(sv)
    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_effective_rank(sv: torch.Tensor):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    :param sv: (float torch Tensor) an array of singular values
    :return: (float torch Tensor) the effective rank
    """
    norm_sv = sv / torch.sum(torch.abs(sv))
    entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * torch.log(p)

    effective_rank = torch.tensor(np.e) ** entropy
    return effective_rank.to(torch.float32)


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper, just that we won't be squaring the singular values
    https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)